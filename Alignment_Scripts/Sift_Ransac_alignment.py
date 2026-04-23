#Ce programme se base sur le code de https://github.com/ily-R/ImageCoregistration/blob/master/image_register.py
"""
Created on Thu Dec 06 10:24:14 2019

@author: ilyas Aroui
"""
import cv2
import numpy as np
import os
from utils import *
import time

DISPLAY_SCALE = 1

base_path = os.path.dirname(__file__)  # dossier du script
img1_path = os.path.join(base_path, "data", "testc1-2vert.png")  # chemin de l'image 1
img2_path = os.path.join(base_path, "data", "testc1-2ir.png")  # chemin de l'image 2

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    print("Erreur : image non trouvee")


def to_single_channel(image, channel_mode):
    if image is None:
        raise ValueError("Image invalide.")

    if image.ndim == 2:
        return image.copy()

    if channel_mode == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channel_mode == "red":
        return image[:, :, 2].copy()
    if channel_mode == "green":
        return image[:, :, 1].copy()
    if channel_mode == "blue":
        return image[:, :, 0].copy()
    if channel_mode == "max":
        return np.max(image, axis=2).astype(image.dtype)

    raise ValueError(f"Canal inconnu: {channel_mode}")


def preprocess_percentile_only(image):
    image_f = image.astype(np.float32)
    p1, p99 = np.percentile(image_f, (1, 99))

    if p99 > p1:
        image_f = np.clip((image_f - p1) / (p99 - p1), 0.0, 1.0)
    else:
        image_f = np.clip(image_f / 255.0, 0.0, 1.0)

    return np.round(image_f * 255.0).astype(np.uint8)


def is_window_open(window_name: str) -> bool:
    return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1


def safe_destroy_window(window_name: str) -> None:
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(window_name)
    except cv2.error:
        pass

def resolve_image_path(image_path):
    if os.path.isabs(image_path) or os.path.exists(image_path):
        return image_path
    return os.path.join("data", image_path)


def readAndRescale(img1, img2, scale, image1_channel="gray", image2_channel="gray", preprocess_mode="none"):
    """Helper to read images, scale them and convert to grayscale.
        it returns original, gray and scaled images

    Typical use:
        t, s, t_gray, s_gray, t_full, s_full = readAndRescale("cat1.jpg", "cat2.jpg", 0.3)

    img1: target image name
    img2: source image name
    scale: scaling factor, keeping aspect ratio
    """
    target = cv2.imread(resolve_image_path(img1), cv2.IMREAD_UNCHANGED)
    source = cv2.imread(resolve_image_path(img2), cv2.IMREAD_UNCHANGED)

    if target is None or source is None:
        raise FileNotFoundError("Impossible de lire une ou plusieurs images dans le dossier data.")

    width = int(target.shape[1] * scale)
    height = int(target.shape[0] * scale)
    dim = (width, height)

    target_s = cv2.resize(target, dim, interpolation=cv2.INTER_AREA)
    source_s = cv2.resize(source, dim, interpolation=cv2.INTER_AREA)

    gray1 = to_single_channel(target_s, image1_channel)
    gray2 = to_single_channel(source_s, image2_channel)

    if preprocess_mode != "none":
        if preprocess_mode == "percentile":
            gray1 = preprocess_percentile_only(gray1)
            gray2 = preprocess_percentile_only(gray2)
        else:
            gray1 = preprocess_multispectral_generic(gray1, mode=preprocess_mode)
            gray2 = preprocess_multispectral_generic(gray2, mode=preprocess_mode)

    return target_s, source_s, gray1, gray2, target, source


def getClicksAndDescriptor(target, source, target_gray, source_gray):
    raise RuntimeError(
        "Le mode manuel a ete desactive dans ce script pour n'afficher que "
        "la fenetre 'Alternating Display'."
    )


def getKeypointAndDescriptors(target_gray, source_gray):
    """Helper to get Harris points of interest and use them as landmarks for sift descriptors.
        it returns these landmarks and their descriptors

    Typical use:
        lmk1, lmk2, desc1, desc2 = getKeypointAndDescriptors(target_gray, source_gray)

    target_gray, source_gray: grayscaled target and source images as np.ndarray
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(target_gray, None)
    pts1 = np.array([kp1[idx].pt for idx in range(len(kp1))])
    kp2, des2 = sift.detectAndCompute(source_gray, None)
    pts2 = np.array([kp2[idx].pt for idx in range(len(kp2))])
    return pts1, pts2, des1, des2, len(kp1), len(kp2)


def display_alternating_images(img_source, img_target, window_name="Alternating Display", delay=1.0):
        """
        Affiche les deux images en alternance dans une seule fenetre.
        
        img_source : np.ndarray
            Image source recalee (warped)
        img_target : np.ndarray
            Image cible (target)
        window_name : str
            Nom de la fenetre OpenCV
        delay : float
            Temps en secondes avant de changer d'image
        """

        # Redimensionner les deux images pour qu'elles aient la meme taille
        height = max(img_source.shape[0], img_target.shape[0])
        width = max(img_source.shape[1], img_target.shape[1])
        
        def resize_to_match(img, height, width):
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        
        img_source_resized = resize_to_match(img_source, height, width)
        img_target_resized = resize_to_match(img_target, height, width)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE))

        try:
            while is_window_open(window_name):
                cv2.imshow(window_name, img_source_resized)
                if cv2.waitKey(int(delay * 1000)) == -1 and not is_window_open(window_name):
                    break

                cv2.imshow(window_name, img_target_resized)
                if cv2.waitKey(int(delay * 1000)) == -1 and not is_window_open(window_name):
                    break
        finally:
            safe_destroy_window(window_name)


def run_alignment(
    image1_path,
    image2_path,
    scale=0.1,
    use_sift=True,
    use_ransac=False,
    image1_channel="gray",
    image2_channel="gray",
    preprocess_mode="percentile",
    ratio_test=0.8,
):
    use_sift = True

    target, source, target_gray, source_gray, target_full, source_full = readAndRescale(
        image1_path,
        image2_path,
        scale,
        image1_channel=image1_channel,
        image2_channel=image2_channel,
        preprocess_mode=preprocess_mode,
    )

    lmk1, lmk2, desc1, desc2, kp1_count, kp2_count = getKeypointAndDescriptors(target_gray, source_gray)
    print(f"Keypoints SIFT detectes: cible={kp1_count}, source={kp2_count}")

    lmk1, lmk2 = match_flann(lmk1, lmk2, desc1, desc2, ratio_test=ratio_test)
    if len(lmk1) == 0 or len(lmk2) == 0:
        raise RuntimeError(
            "Aucun match trouve. Essaie un autre pretraitement ou change les canaux utilises."
        )

    print(f"Nombre de correspondances apres FLANN: {len(lmk1)}")

    if use_ransac:
        if len(lmk1) < 4 or len(lmk2) < 4:
            print("RANSAC ignore: il faut au moins 4 correspondances pour estimer une homographie.")
        else:
            lmk1, lmk2, outliers1, outliers2 = ransac(lmk1, lmk2)
            print(f"Nombre de correspondances conservees apres RANSAC: {len(lmk1)}")

    if len(lmk1) < 3 or len(lmk2) < 3:
        raise RuntimeError(
            "Pas assez de correspondances pour calculer la transformation affine finale."
        )

    T = calculate_transform(lmk2, lmk1)
    warped, target_w = warp(target, source, T)
    cc = cross_corr(warped, target_w)
    print(f"Cross-correlation: {cc}")

    display_alternating_images(warped, target_w, delay=1.0)
    return warped, target_w, T, cc


# -------------------------------------------------------------------------
# ------------------------- IMPLEMENTATION  -------------------------------
# -------------------------------------------------------------------------

if __name__ == "__main__":
    image1_path = "Alignment_Scripts\Image_examples\City1.jpg"
    image2_path = "Alignment_Scripts\Image_examples\City2.jpg"

    scale = 1
    use_sift = True
    use_ransac = False
    image1_channel = "gray"
    image2_channel = "gray"
    preprocess_mode = "percentile"
    ratio_test = 0.8

    warped, target_w, T, cc = run_alignment(
        image1_path=image1_path,
        image2_path=image2_path,
        scale=scale,
        use_sift=use_sift,
        use_ransac=use_ransac,
        image1_channel=image1_channel,
        image2_channel=image2_channel,
        preprocess_mode=preprocess_mode,
        ratio_test=ratio_test,
    )
