#ce code vient de https://github.com/ily-R/ImageCoregistration/blob/master/utils.py
"""
Created on Thu Dec 06 10:24:14 2019

@author: ilyas Aroui
"""

import cv2
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def display_matches(img1, img2, kp1, kp2,name, num=20, save=False):
    """Helper to display matches of keypoint in botch images, by connecting a line from one image to another

    Typical use:
        display_matches(target, source, lmk1, lmk2, name="matches", save = True)

    img1, img2: target and source images as np.ndarray
    kp1, kp2: landmarks of target and source images respectively as np.ndarray
    name: name of the figure display and the image saved if save = True
    save: boolean indicates to save the image of the matches
    """
    if img1.shape[0] != img2.shape[0]:
        minn = min(img1.shape[0], img1.shape[0])
        if minn == img1.shape[0]:
            img1 = np.concatenate((img1, np.zeros(img2.shape[0] - minn, img1.shape[1], 3)), axis=0)
        else:
            img2 = np.concatenate((img2, np.zeros(img1.shape[0] - minn, img2.shape[1], 3)), axis=0)
    img = np.concatenate((img1, img2), axis=1)
    for i in np.random.choice(len(kp1), min(num, len(kp1))):
        x1, y1 = int(kp1[i][0]), int(kp1[i][1])
        x2, y2 = int(kp2[i][0]) + img1.shape[1], int(kp2[i][1])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, img.shape[1], img.shape[0])
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    if save:
        cv2.imwrite(os.path.join("result", name+".jpg"), img)


def preprocess_multispectral_generic(image: np.ndarray, mode: str) -> np.ndarray:
    image_f = image.astype(np.float32)
    p1, p99 = np.percentile(image_f, (1, 99))

    if p99 > p1:
        image_f = np.clip((image_f - p1) / (p99 - p1), 0.0, 1.0)
    else:
        image_f = np.clip(image_f / 255.0, 0.0, 1.0)

    normalized = np.round(image_f * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(normalized)

    if mode == "equalized":
        return equalized
    if mode == "gradient":
        grad_x = cv2.Sobel(equalized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(equalized, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.GaussianBlur(magnitude, (0, 0), 0.8)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if mode == "laplacian":
        laplacian = np.abs(cv2.Laplacian(equalized, cv2.CV_32F, ksize=3))
        return cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if mode == "canny":
        median = float(np.median(equalized))
        low = int(max(0, 0.66 * median))
        high = int(min(255, 1.33 * median))
        return cv2.Canny(equalized, low, high)

    raise ValueError(f"Mode de pretraitement inconnu: {mode}")

def match_flann(lmk1, lmk2, desc1, desc2, ratio_test=0.7):
    """
    FLANN-based matcher pour trouver les correspondances entre deux ensembles de keypoints SIFT.
    lmk1, lmk2 : keypoints (np.ndarray Nx2)
    desc1, desc2 : descripteurs SIFT (np.ndarray Nx128)
    ratio_test : ratio de Lowe pour filtrer les mauvais matches
    """
    if desc1 is None or desc2 is None:
        return np.array([]), np.array([])

    if len(desc1) < 2 or len(desc2) < 2:
        return np.array([]), np.array([])

    # Paramètres FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # knnMatch pour trouver les 2 meilleurs matches pour chaque point
    matches = flann.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
    
    good_lmk1 = []
    good_lmk2 = []

    # Ratio test de Lowe
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good_lmk1.append(lmk1[m.queryIdx])
            good_lmk2.append(lmk2[m.trainIdx])

    return np.array(good_lmk1), np.array(good_lmk2)

def cross_corr(img1, img2, show_images=True):
    """
    Calcule la cross-correlation entre deux images en ignorant les pixels noirs.

    Parameters
    ----------
    img1 : np.ndarray
        Première image (BGR ou grayscale).
    img2 : np.ndarray
        Deuxième image (BGR ou grayscale).
    show_images : bool
        Si True, affiche les images utilisées pour le calcul.

    Returns
    -------
    corr : float
        Valeur de la cross-correlation.
    """
    # Convertir en grayscale si nécessaire
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1.copy()

    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2.copy()

    # Masque pour ignorer les pixels noirs dans au moins une des deux images
    mask = (img1_gray > 0) & (img2_gray > 0)

    # Affichage des images utilisées
    #if show_images:
     #   display1 = img1_gray.copy()
      #  display2 = img2_gray.copy()
       # display1[~mask] = 0
        #display2[~mask] = 0
        #cv2.imshow("Image1 used for CC", display1)
        #cv2.imshow("Image2 used for CC", display2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    # Centrer les images sur zéro (moyenne)
    img1_c = img1_gray.astype(np.float32) - np.mean(img1_gray[mask])
    img2_c = img2_gray.astype(np.float32) - np.mean(img2_gray[mask])

    # Calcul de la cross-correlation sur les pixels valides
    numerator = np.sum(img1_c[mask] * img2_c[mask])
    denominator = np.sqrt(np.sum(img1_c[mask] ** 2) * np.sum(img2_c[mask] ** 2))
    corr = numerator / denominator

    print("Cross-correlation (ignoring black pixels):", corr)
    return corr


def mutual_inf(img1, img2, verbose=False):
    """Helper to calculate mutual-information metric between two images. it gives a probabilistic measure on how
    uncertain we are about the target image in the absence/presence of the warped source image
    it returns the mutual information value.

    Typical use:
        mi = mutual_inf(warped, target_w)

    verbose: if verbose=True, display and save the joint-histogram between the two images.
    """
    epsilon = 1.e-6
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = np.round(img1).astype("uint8")
    img2 = np.round(img2).astype("uint8")

    joint_hist = np.zeros((256, 256))
    for i in range(min(img1.shape[0], img2.shape[0])):
        for j in range(min(img1.shape[1], img2.shape[1])):
            joint_hist[img1[i, j], img2[i, j]] += 1

    if verbose:
        display_jh = np.log(joint_hist + epsilon)
        display_jh = 255*(display_jh - display_jh.min())/(display_jh.max() - display_jh.min())
        display_jh_resized = cv2.resize(display_jh, (800, 800), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("joint_histogram", display_jh_resized)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.imwrite("result/joint_histogram.jpg", display_jh)

    joint_hist /= np.sum(joint_hist)
    p1 = np.sum(joint_hist, axis=0)
    p2 = np.sum(joint_hist, axis=1)
    joint_hist_d = joint_hist/(p1+epsilon)
    joint_hist_d /= (p2+epsilon)
    mi = np.sum(np.multiply(joint_hist, np.log(joint_hist_d+epsilon)))
    print("Mutual Information: ", mi)
    return mi


def ransac(kp1, kp2):
    if len(kp1) < 4 or len(kp2) < 4:
        return kp1, kp2, np.empty((0, 2)), np.empty((0, 2))
    H, mask = cv2.findHomography(kp1, kp2, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return kp1, kp2, np.empty((0, 2)), np.empty((0, 2))
    inliers1 = kp1[mask.ravel() == 1]
    inliers2 = kp2[mask.ravel() == 1]
    outliers1 = kp1[mask.ravel() == 0]
    outliers2 = kp2[mask.ravel() == 0]
    return inliers1, inliers2, outliers1, outliers2


def calculate_transform(kp1, kp2):
    """Helper to apply find the best affine transform using two arrays of landmarks.
    it returns the affine transform, a matrix T of size (2, 3)

    Typical use:
        T = calculate_transform(lmk2, lmk1)

    kp1, kp2: landmarks of target and source images respectively as np.ndarray
    """
    upper = np.concatenate((kp1, np.ones((kp1.shape[0], 1)), np.zeros((kp1.shape[0], 3))), axis=1)
    lower = np.concatenate((np.zeros((kp1.shape[0], 3)), kp1, np.ones((kp1.shape[0], 1))), axis=1)
    X = np.concatenate((upper, lower), axis=0)
    Y = np.concatenate((kp2[:, 0], kp2[:, 1]))
    Y = np.expand_dims(Y, axis=-1)
    T = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    T = T.reshape(2, 3)
    T = np.concatenate((T, np.array([0, 0, 1]).reshape(1, 3)), axis=0)
    kp2_pred = np.dot(T, np.concatenate((kp1, np.ones((kp1.shape[0], 1))), axis=1).T).T
    kp2_pred /= kp2_pred[:, -1:]
    error = np.linalg.norm(kp2_pred[:, :2] - kp2)
    print("coordinate reconstruction error: ", error)
    return T


def warp(target, source, T):

    """Helper to move the source image to the same reference as target image, so they can be co-registered.
    it returns the new warped source image and the target image which is also centered in a larger figure by 10 pixels.
    i.e, if the input size is (M, N) then the output is (M+10, N+10).

    Typical use:
        warped, target_w = warp(target, source, T)

    T:  affine transform, a matrix T of size (2, 3)
    """
    height = target.shape[0]
    width = source.shape[1]

    # move both images to the center a bit
    corners = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    corners_moved = np.float32([[5, 5], [5, height + 5], [5 + width, 5], [5 + width, 5 + height]])
    T_perspective = cv2.getPerspectiveTransform(corners, corners_moved)
    target_new = cv2.warpPerspective(target, T_perspective, (width + 10, height + 10))
    cv2.imshow("target_new", target_new)
    cv2.imwrite("result/target_new.jpg", target_new)
    T = np.dot(T_perspective, T)
    source_new = cv2.warpPerspective(source, T, (width + 10, height + 10), cv2.INTER_AREA)
    cv2.imshow("source_new", source_new)  # show transform
    cv2.imwrite("result/source_new.jpg", source_new)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return source_new, target_new