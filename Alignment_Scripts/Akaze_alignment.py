import os
import cv2
import numpy as np

scale = 0.5
RESULT_DIR = "result"


def is_window_open(window_name: str) -> bool:
    return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1


def wait_until_window_closed(window_name: str, delay_ms: int = 50) -> None:
    while is_window_open(window_name):
        cv2.waitKey(delay_ms)


def safe_destroy_window(window_name: str) -> None:
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def ensure_result_dir() -> None:
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)


def preprocess_multispectral(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32)

    p1, p99 = np.percentile(image_f, (1, 99))
    if p99 > p1:
        image_f = np.clip((image_f - p1) / (p99 - p1), 0.0, 1.0)
    else:
        image_f = np.clip(image_f / 255.0, 0.0, 1.0)

    image_u8 = np.round(image_f * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    equalized = clahe.apply(image_u8)

    grad_x = cv2.Sobel(equalized, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(equalized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.GaussianBlur(magnitude, (0, 0), 0.8)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def mutual_inf(img1, img2, verbose=False):
    epsilon = 1e-6

    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1.copy()

    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2.copy()

    img1_gray = np.round(img1_gray).astype(np.uint8)
    img2_gray = np.round(img2_gray).astype(np.uint8)

    h = min(img1_gray.shape[0], img2_gray.shape[0])
    w = min(img1_gray.shape[1], img2_gray.shape[1])
    joint_hist = np.zeros((256, 256), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            joint_hist[img1_gray[i, j], img2_gray[i, j]] += 1

    if verbose:
        display_jh = np.log(joint_hist + epsilon)
        display_jh = 255 * (display_jh - display_jh.min()) / (display_jh.max() - display_jh.min())
        display_jh_resized = cv2.resize(display_jh, (800, 800), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("joint_histogram", display_jh_resized.astype(np.uint8))
        wait_until_window_closed("joint_histogram")
        safe_destroy_window("joint_histogram")
        ensure_result_dir()
        cv2.imwrite(os.path.join(RESULT_DIR, "joint_histogram.jpg"), display_jh_resized.astype(np.uint8))

    joint_hist /= np.sum(joint_hist)
    p1 = np.sum(joint_hist, axis=1)
    p2 = np.sum(joint_hist, axis=0)
    joint_hist_d = joint_hist / (p1[:, None] + epsilon)
    joint_hist_d /= (p2[None, :] + epsilon)
    mi = np.sum(joint_hist * np.log(joint_hist_d + epsilon))

    print("Mutual Information:", mi)
    return mi


def display_alternating_images(img_ref, img_corrected, window_name="Alternating Display", delay=1.0):
    if len(img_ref.shape) == 2:
        img_ref_disp = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)
    else:
        img_ref_disp = img_ref.copy()

    if len(img_corrected.shape) == 2:
        img_corr_disp = cv2.cvtColor(img_corrected, cv2.COLOR_GRAY2BGR)
    else:
        img_corr_disp = img_corrected.copy()

    height = max(img_ref_disp.shape[0], img_corr_disp.shape[0])
    width = max(img_ref_disp.shape[1], img_corr_disp.shape[1])

    img_ref_resized = cv2.resize(img_ref_disp, (width, height), interpolation=cv2.INTER_LINEAR)
    img_corr_resized = cv2.resize(img_corr_disp, (width, height), interpolation=cv2.INTER_LINEAR)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width*2, height*2)

    try:
        while is_window_open(window_name):
            cv2.imshow(window_name, img_ref_resized)
            if cv2.waitKey(int(delay * 1000)) == -1 and not is_window_open(window_name):
                break

            cv2.imshow(window_name, img_corr_resized)
            if cv2.waitKey(int(delay * 1000)) == -1 and not is_window_open(window_name):
                break
    finally:
        safe_destroy_window(window_name)


def masked_cross_correlation(img1_gray, img2_gray):
    mask = (img1_gray > 0) & (img2_gray > 0)

    if np.sum(mask) == 0:
        return 0

    img1_f = img1_gray.astype(np.float32)
    img2_f = img2_gray.astype(np.float32)

    mean1 = np.mean(img1_f[mask])
    mean2 = np.mean(img2_f[mask])

    img1_c = img1_f - mean1
    img2_c = img2_f - mean2

    numerator = np.sum(img1_c[mask] * img2_c[mask])
    denominator = np.sqrt(np.sum(img1_c[mask] ** 2) * np.sum(img2_c[mask] ** 2))

    if denominator == 0:
        return 0

    return numerator / denominator


# -------------------------------------------------------------------------
# ------------------------- IMPLEMENTATION  -------------------------------
# -------------------------------------------------------------------------

if __name__ == "__main__":

    I1 = cv2.imread("Alignment_Scripts\Image_examples\City1.jpg", cv2.IMREAD_GRAYSCALE)
    I2 = cv2.imread("Alignment_Scripts\Image_examples\City2.jpg", cv2.IMREAD_GRAYSCALE)

    if I1 is None or I2 is None:
        raise FileNotFoundError("Impossible de lire les images data/grillevert_modif.jpg et/ou data/grilleIR_modif.jpg")

    I1 = cv2.resize(I1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    I2 = cv2.resize(I2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    I1_proc = preprocess_multispectral(I1)
    I2_proc = preprocess_multispectral(I2)

    akaze = cv2.AKAZE_create(threshold=1e-4)
    kpts1, desc1 = akaze.detectAndCompute(I1_proc, None)
    kpts2, desc2 = akaze.detectAndCompute(I2_proc, None)

    img_kp1 = cv2.drawKeypoints(I1_proc, kpts1, None)
    img_kp2 = cv2.drawKeypoints(I2_proc, kpts2, None)

    print("Number of  keypoints in image 1 :", len(kpts1))
    print("Number of  keypoints in image 2 :", len(kpts2))

    if desc1 is None or desc2 is None:
        raise RuntimeError("AKAZE n'a pas trouve suffisamment de descripteurs apres pretraitement.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches_21 = bf.knnMatch(desc2, desc1, k=2)
    knn_matches_12 = bf.knnMatch(desc1, desc2, k=2)

    forward_best = {}
    for m, n in knn_matches_21:
        if m.distance < 0.85 * n.distance:
            forward_best[(m.queryIdx, m.trainIdx)] = m

    good_matches = []
    for m, n in knn_matches_12:
        if m.distance < 0.85 * n.distance and (m.trainIdx, m.queryIdx) in forward_best:
            good_matches.append(m)

    print("Number of good correspondences :", len(good_matches))

    match_img = cv2.drawMatches(I1_proc, kpts1, I2_proc, kpts2, good_matches, None)
    cv2.imshow("Matches", match_img)
    ensure_result_dir()
    cv2.imwrite(os.path.join(RESULT_DIR, "Matches.jpg"), match_img)
    wait_until_window_closed("Matches")
    safe_destroy_window("Matches")

    if len(good_matches) < 4:
        raise RuntimeError("Pas assez de bonnes correspondances pour estimer une homographie.")

    pts1 = np.float32([kpts1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kpts2[m.trainIdx].pt for m in good_matches])

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Echec de l'estimation de l'homographie.")

    h, w = I1.shape
    I2_corrected = cv2.warpPerspective(I2, H, (w, h))

    display_alternating_images(I1, I2_corrected, delay=1.0)
    mi_value = mutual_inf(I1, I2_corrected, verbose=True)

    cv2.imshow("Image 2 corrigee", I2_corrected)
    ensure_result_dir()
    cv2.imwrite(os.path.join(RESULT_DIR, "I2_corrected.jpg"), I2_corrected)

    correlation = masked_cross_correlation(I1, I2_corrected)
    print("Cross-correlation :", correlation)

    print('Results have been saved in result folder')
    wait_until_window_closed("Image 2 corrigee")
    safe_destroy_window("Image 2 corrigee")
    
