import cv2
import numpy as np

DISPLAY_SCALE = 0.2


def preprocess_for_ecc(image: np.ndarray) -> np.ndarray:
    image_u8 = image.astype(np.uint8)
    image_eq = cv2.equalizeHist(image_u8)
    image_blur = cv2.GaussianBlur(image_eq, (5, 5), 0)
    return image_blur.astype(np.float32) / 255.0


def ecc_registration(ref_img, mov_img):

    """
    Register two images using the Enhanced Correlation Coefficient (ECC) algorithm.

    Parameters
    ----------
    ref_img : numpy.ndarray
        Reference image.
    mov_img : numpy.ndarray
        Image to be registered.

    Returns
    -------
    aligned : numpy.ndarray
        Registered image.
    warp_matrix : numpy.ndarray
        Affine transformation matrix.

    Notes
    -----
    The ECC algorithm is a robust and efficient method for registration of two images.
    It is based on the maximization of the cross-correlation between the two images.
    """
    if ref_img.shape != mov_img.shape:
        mov_img = cv2.resize(
            mov_img,
            (ref_img.shape[1], ref_img.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    ref_proc = preprocess_for_ecc(ref_img)
    mov_proc = preprocess_for_ecc(mov_img)

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        200,
        1e-6
    )

    motion_models = [cv2.MOTION_EUCLIDEAN, cv2.MOTION_TRANSLATION]
    last_error = None

    for motion_model in motion_models:
        try:
            _, warp_matrix = cv2.findTransformECC(
                ref_proc,
                mov_proc,
                warp_matrix.copy(),
                motion_model,
                criteria
            )
            break
        except cv2.error as exc:
            last_error = exc
    else:
        raise RuntimeError(
            "ECC n'a pas converge. Les images sont peut-etre trop differentes, "
            "trop peu recouvrantes, ou necessitent un autre mode de recalage."
        ) from last_error

    height, width = ref_img.shape

    aligned = cv2.warpAffine(
        mov_img,
        warp_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )

    return aligned, warp_matrix


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


def display_alternating_images(
    img_ref: np.ndarray,
    img_aligned: np.ndarray,
    window_name: str = "Comparaison alignement",
    delay: float = 1.0,
    scale: float = DISPLAY_SCALE,
) -> None:
    height = max(img_ref.shape[0], img_aligned.shape[0])
    width = max(img_ref.shape[1], img_aligned.shape[1])

    ref_resized = cv2.resize(img_ref, (width, height), interpolation=cv2.INTER_LINEAR)
    aligned_resized = cv2.resize(img_aligned, (width, height), interpolation=cv2.INTER_LINEAR)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(width * scale), int(height * scale))

    try:
        while is_window_open(window_name):
            cv2.imshow(window_name, ref_resized)
            if cv2.waitKey(int(delay * 1000)) == -1 and not is_window_open(window_name):
                break

            cv2.imshow(window_name, aligned_resized)
            if cv2.waitKey(int(delay * 1000)) == -1 and not is_window_open(window_name):
                break
    finally:
        safe_destroy_window(window_name)


# -------------------------------------------------------------------------
# ------------------------- IMPLEMENTATION  -------------------------------
# -------------------------------------------------------------------------

if __name__ == "__main__":
    ref_path = "Alignment_Scripts\Image_examples\ShippingDock1.jpg"
    mov_path = "Alignment_Scripts\Image_examples\ShippingDock2.jpg"

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    mov_img = cv2.imread(mov_path, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or mov_img is None:
        raise FileNotFoundError(
            f"Impossible de lire les images : '{ref_path}' et/ou '{mov_path}'"
        )

    aligned_img, warp_matrix = ecc_registration(ref_img, mov_img)

    print("Matrice de transformation :")
    print(warp_matrix)

    display_alternating_images(ref_img, aligned_img, delay=1.0)
