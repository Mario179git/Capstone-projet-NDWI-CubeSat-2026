import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import os
from datetime import datetime
import cv2
from picamera2 import Picamera2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from alignment_akaze import preprocess_multispectral
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# ============================================================
# Fenêtre
# ============================================================

root = tk.Tk()
root.title("Interface Caméras Duo - Raspberry Pi 5")
root.geometry("1600x800")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

DEFAULT_NDWI_SAVE_PATH = "/home/capstone/Desktop/NDWI_graphs_saved"
DEFAULT_CAMERA_SAVE_PATH = "/home/capstone/Desktop/Individual_camera_images_saved"

save_path = DEFAULT_NDWI_SAVE_PATH
video_running = False
# ============================================================
# Image noire utilitaire (avant qu'une image soit prise)
# ============================================================

def create_black_image(width=640, height=480, text=None):
    img = Image.new("L", (width, height), 0)
    if text:
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0,0), text)
        tw = bbox[2]-bbox[0]
        th = bbox[3]-bbox[1]
        draw.text(((width-tw)//2,(height-th)//2), text, fill=255)
    return img

# ============================================================
# Onglet NDWI
# ============================================================

last_ndwi_fig = None

def snapshot():
    global last_ndwi_fig

    if not cameras_ok:
        print("Caméras non disponibles")
        return

    try:
        frame0 = picam2_0.capture_array()
        frame1 = picam2_1.capture_array()
        
        frame0 = cv2.rotate(frame0, cv2.ROTATE_180)
        frame1 = cv2.rotate(frame1, cv2.ROTATE_180)

        nir = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        green = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)

        try:
            scale = 0.5

            I1 = cv2.resize(nir, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            I2 = cv2.resize(green, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            I1_proc = preprocess_multispectral(I1)
            I2_proc = preprocess_multispectral(I2)

            akaze = cv2.AKAZE_create(threshold=1e-4)
            kpts1, desc1 = akaze.detectAndCompute(I1_proc, None)
            kpts2, desc2 = akaze.detectAndCompute(I2_proc, None)

            if desc1 is None or desc2 is None:
                raise RuntimeError("Pas assez de descripteurs")

            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(desc1, desc2, k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.85 * n.distance:
                    good.append(m)

            if len(good) < 4:
                raise RuntimeError("Pas assez de matches")

            pts1 = np.float32([kpts1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kpts2[m.trainIdx].pt for m in good])

            H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

            h, w = I1.shape
            I2_corrected = cv2.warpPerspective(I2, H, (w, h))

            nir = I1
            green = I2_corrected


            # --- masque circulaire ---
            img_blur = cv2.GaussianBlur(nir, (9, 9), 2)
            _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cnt = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(cnt)

            center = (int(x), int(y))
            radius = int(radius * 1.4)

            circle_mask = np.zeros_like(nir, dtype=np.uint8)
            cv2.circle(circle_mask, center, radius, 1, -1)

        except Exception as e:
            print("Alignement échoué :", e)
            return

        # --- NDWI ---
        nir = nir.astype(np.float32)
        green = green.astype(np.float32)

        eps = 1e-5
        valid_mask = (nir > eps) & (green > eps)

        boost = 0.97
        num = green - nir * boost
        den = green + nir * boost + 1e-6

        ndwi = np.full_like(den, np.nan, dtype=np.float32)
        ndwi_temp = num / den

        final_mask = (circle_mask == 1) & valid_mask
        ndwi[final_mask] = ndwi_temp[final_mask]

        # --- LISSAGE CORRECT (anti-NaN) ---
        sigma = 1

        ndwi_filled = np.nan_to_num(ndwi, nan=0.0)
        ndwi_blur = gaussian_filter(ndwi_filled, sigma=sigma)

        mask = (~np.isnan(ndwi)).astype(float)
        mask_blur = gaussian_filter(mask, sigma=sigma)

        ndwi_smooth = ndwi_blur / (mask_blur + 1e-6)
        ndwi_smooth[mask_blur < 1e-3] = np.nan

        # --- CROP SANS AUCUN PIXEL NOIR (NaN) ---

        valid_mask = ~np.isnan(ndwi_smooth)

        h, w = ndwi_smooth.shape

        # centre basé sur les pixels valides
        coords = np.column_stack(np.where(valid_mask))

        if coords.size > 0:
            cy = int(np.mean(coords[:, 0]))
            cx = int(np.mean(coords[:, 1]))

            max_half = min(cy, cx, h - cy, w - cx)

            best_half = 0

            # on agrandit progressivement le carré
            for half in range(1, max_half):
                y1 = cy - half
                y2 = cy + half
                x1 = cx - half
                x2 = cx + half

                sub_mask = valid_mask[y1:y2, x1:x2]

                # si tout est valide → on garde
                if np.all(sub_mask):
                    best_half = half
                else:
                    break

            # crop final
            y1 = cy - best_half
            y2 = cy + best_half
            x1 = cx - best_half
            x2 = cx + best_half

            ndwi_smooth = ndwi_smooth[y1:y2, x1:x2]

        # --- AFFICHAGE ---
        colors = [(0.0, "green"), (0.5, "white"), (1.0, "blue")]
        custom_cmap = LinearSegmentedColormap.from_list("GreenWhiteBlue", colors)
        custom_cmap.set_bad(color='black')

        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

        im = ax.imshow(ndwi_smooth, cmap=custom_cmap, vmin=-1, vmax=1)
        ax.set_title("NDWI")
        fig.colorbar(im, ax=ax)

        canvas1.delete("all")

        canvas_fig = FigureCanvasTkAgg(fig, master=canvas1)
        canvas_fig.draw()
        widget = canvas_fig.get_tk_widget()
        widget.place(relx=0.5, rely=0.5, anchor="center")

        last_ndwi_fig = fig
        plt.close(fig)

    except Exception as e:
        print("Erreur snapshot :", e)

def select_path():
    global save_path
    folder = filedialog.askdirectory()
    if folder:
        save_path = folder
        path_label.config(text=f"Save path: {save_path}")

def save_current_image():
    global save_path, last_ndwi_fig

    if save_path is None:
        print("Aucun chemin sélectionné")
        return

    if last_ndwi_fig is None:
        print("Aucun graphique NDWI à sauvegarder")
        return

    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(save_path, f"NDWI_{timestamp}.png")
    last_ndwi_fig.savefig(filepath, dpi=300)
    print("Graphique NDWI sauvegardé :", filepath)


tab1 = ttk.Frame(notebook)
notebook.add(tab1,text="Main: ndwi")

canvas1 = tk.Canvas(tab1,bg="black")
canvas1.pack(fill="both",expand=True)

def draw_placeholder(canvas):
    canvas.delete("all")
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    canvas.create_rectangle(0,0,w,h,fill="black")
    canvas.create_text(
        w//2, h//2,
        text="Snapshot pour afficher le graphique",
        fill="white",
        font=("Arial",32)
    )

canvas1.bind("<Configure>",lambda e: draw_placeholder(canvas1))

button_frame = tk.Frame(tab1)
button_frame.pack(side="bottom",pady=20)

path_label = tk.Label(button_frame, text=f"Save path: {save_path}")
path_label.pack()

tk.Button(button_frame,text="Snapshot",command=snapshot).pack(side="left",padx=10)
tk.Button(button_frame,text="Select Path",command=select_path).pack(side="left",padx=10)
tk.Button(button_frame,text="Save Image",command=save_current_image).pack(side="left",padx=10)

# ============================================================
# Onglet Caméras
# ============================================================

tab2 = ttk.Frame(notebook)
notebook.add(tab2,text="Camera feeds")

frame_video = tk.Frame(tab2)
frame_video.pack(side="left",fill="both",expand=True)

frame_settings = tk.LabelFrame(tab2,text="Parameters")
frame_settings.pack(side="right",fill="y",padx=10,pady=10)

video_label = tk.Label(frame_video,bg="black")
video_label.pack(fill="both",expand=True)

# ============================================================
# FPS + Exposition
# ============================================================

fps_var = tk.IntVar(value=1)
tk.Label(frame_settings, text="FPS").pack(anchor="w")
fps_entry = tk.Entry(frame_settings, textvariable=fps_var, width=10)
fps_entry.pack(anchor="w")

tk.Label(frame_settings,text="Exposure (µs)").pack(anchor="w")
exposure_var = tk.IntVar(value=200000)

exposure_slider = tk.Scale(
    frame_settings,
    from_=50,
    to=500000,
    orient="horizontal",
    variable=exposure_var,
    length=200
)
exposure_slider.pack(anchor="w")


def apply_camera_settings():
    try:
        fps = int(fps_entry.get())
    except ValueError:
        print("FPS invalide !")
        return False

    fps = max(1, min(30, fps))
    fps_var.set(fps)

    exp = max(50, min(500000, int(exposure_var.get())))
    exposure_var.set(exp)

    frame_duration = int(1_000_000 / fps)

    if cameras_ok:
        picam2_0.set_controls({'FrameDurationLimits': (frame_duration, frame_duration)})
        picam2_1.set_controls({'FrameDurationLimits': (frame_duration, frame_duration)})

        picam2_0.set_controls({"ExposureTime": exp})
        picam2_1.set_controls({"ExposureTime": exp})

        print(f"FPS appliqué : {fps} | Exposition appliquée : {exp} µs")

    return True

# ============================================================
# Initialisation caméras
# ============================================================

try:
    picam2_0 = Picamera2(0)
    picam2_1 = Picamera2(1)

    w, h = 1600, 1300

    config0 = picam2_0.create_preview_configuration(
        main={"size":(w,h),"format":"RGB888"}
    )
    config1 = picam2_1.create_preview_configuration(
        main={"size":(w,h),"format":"RGB888"}
    )

    picam2_0.configure(config0)
    picam2_1.configure(config1)

    picam2_0.start()
    picam2_1.start()

    picam2_0.set_controls({"AeEnable": False})
    picam2_1.set_controls({"AeEnable": False})

    cameras_ok = True
    apply_camera_settings()
    print("Caméras initialisées")

except Exception as e:
    print("Erreur caméra :",e)
    cameras_ok = False

# ============================================================
# Bouton OK : applique FPS + Exposition
# ============================================================

def apply_fps():
    apply_camera_settings()

tk.Button(frame_settings, text="OK", command=apply_fps).pack(anchor="w", pady=5)

# ============================================================
# Sauvegarde images caméras (images séparées)
# ============================================================

camera_save_path = DEFAULT_CAMERA_SAVE_PATH

def select_camera_path():
    global camera_save_path
    folder = filedialog.askdirectory()
    if folder:
        camera_save_path = folder
        camera_path_label.config(text=f"Save path: {camera_save_path}")

def save_camera_images():
    global camera_save_path

    if camera_save_path is None:
        print("Aucun chemin sélectionné pour les caméras")
        return

    try:
        os.makedirs(camera_save_path, exist_ok=True)
        frame0 = picam2_0.capture_array()
        frame1 = picam2_1.capture_array()

        frame0 = cv2.rotate(frame0, cv2.ROTATE_180)
        frame1 = cv2.rotate(frame1, cv2.ROTATE_180)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path0 = os.path.join(camera_save_path, f"{timestamp}_IR.png")
        path1 = os.path.join(camera_save_path, f"{timestamp}_Vert.png")

        cv2.imwrite(path0, cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))
        cv2.imwrite(path1, cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR))

        print("Images sauvegardées :")
        print(" -", path0)
        print(" -", path1)

    except Exception as e:
        print("Erreur sauvegarde images :", e)

camera_path_label = tk.Label(frame_settings, text=f"Save path: {camera_save_path}")
camera_path_label.pack(anchor="w", pady=(10,0))

tk.Button(frame_settings, text="Select Path", command=select_camera_path).pack(anchor="w", pady=5)
tk.Button(frame_settings, text="Save Images", command=save_camera_images).pack(anchor="w", pady=5)

# ============================================================
# Boucle vidéo
# ============================================================

last_frame = create_black_image(1280,480,"Initialisation...")

def update_video():
    global last_frame, video_running
    if not video_running:
        return

    if cameras_ok:
        try:
            frame0 = picam2_0.capture_array()
            frame1 = picam2_1.capture_array()
            
            frame0 = cv2.flip(frame0, -1)
            frame1 = cv2.flip(frame1, -1)

            frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)

            frame0 = cv2.resize(frame0,(640,480))
            frame1 = cv2.resize(frame1,(640,480))

            combined = cv2.hconcat([frame0,frame1])

            cv2.putText(combined, "860 nm", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

            cv2.putText(combined, "560 nm", (650, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

            last_frame = Image.fromarray(combined)

        except Exception as e:
            print("Erreur capture :",e)

    imgtk = ImageTk.PhotoImage(last_frame)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    root.after(1, update_video)

# ============================================================
# Détection changement d'onglet
# ============================================================

def on_tab_change(event):
    global video_running
    current = notebook.index(notebook.select())
    if current == 1:
        video_running = True
        update_video()
    else:
        video_running = False

notebook.bind("<<NotebookTabChanged>>", on_tab_change)

root.mainloop()
