import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from plot_psf import (
    choose_psf_center_interactively,
    estimate_psf_from_image_file,
    load_grayscale_image,
    parse_local_psf_names,
)


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image: {path}")

    if img.ndim == 2:
        img = img.astype(np.float32) / 255.0
    elif img.ndim == 3:
        img = img.astype(np.float32) / 255.0
    else:
        raise ValueError("Image non supportee (ndim != 2/3).")

    return img


def save_image(path: Path, img: np.ndarray) -> None:
    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(path), out)
    if not ok:
        raise RuntimeError(f"Echec ecriture image: {path}")


def load_psf(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        psf = np.load(str(path)).astype(np.float32)
    else:
        psf = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if psf is None:
            raise FileNotFoundError(f"Impossible de lire la PSF: {path}")
        psf = psf.astype(np.float32)

    if psf.ndim != 2:
        raise ValueError("La PSF doit etre 2D.")

    psf = np.maximum(psf, 0.0)
    s = float(psf.sum())
    if s <= 0:
        raise ValueError("PSF invalide: somme <= 0.")

    psf /= s
    return psf


def to_grayscale_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    gray_u8 = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return gray_u8.astype(np.float32) / 255.0


def build_useful_region_mask(
    image: np.ndarray,
    threshold: int = 11,
    close_radius: int = 9,
    margin_ratio: float = 1.02,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], float]:
    gray_u8 = np.clip(to_grayscale_float(image) * 255.0, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(gray_u8, threshold, 255, cv2.THRESH_BINARY)

    kernel_size = max(3, close_radius)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Aucune zone utile detectee pour la deconvolution locale.")

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        raise ValueError("Contour insuffisant pour ajuster une ellipse sur la zone utile.")

    (cx, cy), (major_axis, minor_axis), angle = cv2.fitEllipse(largest)
    axes = (
        max(1.0, major_axis * margin_ratio / 2.0),
        max(1.0, minor_axis * margin_ratio / 2.0),
    )

    mask = np.zeros(gray_u8.shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(cx)), int(round(cy))),
        (int(round(axes[0])), int(round(axes[1]))),
        angle,
        0,
        360,
        255,
        -1,
    )
    return mask, (cx, cy), axes, angle


def _find_axis_span(mask: np.ndarray, center_index: int, axis: int, band_half_width: int = 3) -> tuple[int, int]:
    if axis == 0:
        start = max(0, center_index - band_half_width)
        end = min(mask.shape[1], center_index + band_half_width + 1)
        profile = np.any(mask[:, start:end] > 0, axis=1)
    else:
        start = max(0, center_index - band_half_width)
        end = min(mask.shape[0], center_index + band_half_width + 1)
        profile = np.any(mask[start:end, :] > 0, axis=0)

    indices = np.flatnonzero(profile)
    if indices.size == 0:
        raise ValueError("Impossible de trouver les bords de la zone utile sur l'axe central.")
    return int(indices[0]), int(indices[-1])


def compute_local_anchor_points(
    mask: np.ndarray,
) -> dict[str, tuple[float, float]]:
    yy, xx = np.nonzero(mask > 0)
    if xx.size == 0 or yy.size == 0:
        raise ValueError("Le masque de zone utile est vide.")

    cx = float(np.mean(xx))
    cy = float(np.mean(yy))

    center_x = int(round(cx))
    center_y = int(round(cy))

    top_y, bottom_y = _find_axis_span(mask, center_x, axis=0)
    left_x, right_x = _find_axis_span(mask, center_y, axis=1)

    return {
        "centre": (cx, cy),
        "haut": (cx, float(top_y)),
        "bas": (cx, float(bottom_y)),
        "gauche": (float(left_x), cy),
        "droite": (float(right_x), cy),
    }


def compute_local_region_masks(
    mask: np.ndarray,
    anchors: dict[str, tuple[float, float]],
    center_radius: float = 0.35,
    feather_sigma: float = 9.0,
) -> dict[str, np.ndarray]:
    cx, cy = anchors["centre"]
    left_x, _ = anchors["gauche"]
    right_x, _ = anchors["droite"]
    _, top_y = anchors["haut"]
    _, bottom_y = anchors["bas"]

    yy, xx = np.indices(mask.shape, dtype=np.float32)

    dx = xx - cx
    dy = yy - cy

    rx = np.where(dx >= 0.0, max(1.0, right_x - cx), max(1.0, cx - left_x))
    ry = np.where(dy >= 0.0, max(1.0, bottom_y - cy), max(1.0, cy - top_y))

    xn = dx / rx
    yn = dy / ry
    radius2 = xn**2 + yn**2

    valid = mask > 0
    center_region = valid & (radius2 <= center_radius**2)
    outer = valid & ~center_region

    top_region = outer & (np.abs(yn) >= np.abs(xn)) & (yn < 0)
    bottom_region = outer & (np.abs(yn) >= np.abs(xn)) & (yn >= 0)
    left_region = outer & (np.abs(xn) > np.abs(yn)) & (xn < 0)
    right_region = outer & (np.abs(xn) > np.abs(yn)) & (xn >= 0)

    region_masks = {
        "centre": center_region.astype(np.float32),
        "haut": top_region.astype(np.float32),
        "bas": bottom_region.astype(np.float32),
        "gauche": left_region.astype(np.float32),
        "droite": right_region.astype(np.float32),
    }

    if feather_sigma > 0:
        for name, region in list(region_masks.items()):
            blurred = cv2.GaussianBlur(region, (0, 0), feather_sigma)
            blurred[~valid] = 0.0
            region_masks[name] = blurred

    total = np.zeros(mask.shape, dtype=np.float32)
    for region in region_masks.values():
        total += region
    total = np.maximum(total, 1e-6)

    for name in list(region_masks):
        normalized = region_masks[name] / total
        normalized[~valid] = 0.0
        region_masks[name] = normalized

    return region_masks


def _fft_kernel(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.fft.fft2(np.fft.ifftshift(kernel), s=shape)


def fft_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kf = _fft_kernel(kernel, image.shape)
    out = np.fft.ifft2(np.fft.fft2(image) * kf).real
    return out


def apply_aberration_with_psf(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return fft_convolve2d(image, psf)

    channels = []
    for c in range(image.shape[2]):
        channels.append(fft_convolve2d(image[:, :, c], psf))
    return np.stack(channels, axis=2)


def wiener_deconvolution_channel(channel: np.ndarray, psf: np.ndarray, balance: float) -> np.ndarray:
    h = _fft_kernel(psf, channel.shape)
    g = np.fft.fft2(channel)
    h_conj = np.conj(h)
    f_hat = (h_conj / (np.abs(h) ** 2 + balance)) * g
    out = np.fft.ifft2(f_hat).real
    return np.clip(out, 0.0, 1.0)


def wiener_deconvolution(image: np.ndarray, psf: np.ndarray, balance: float = 1e-3) -> np.ndarray:
    if image.ndim == 2:
        return wiener_deconvolution_channel(image, psf, balance)

    out_channels = []
    for c in range(image.shape[2]):
        out_channels.append(wiener_deconvolution_channel(image[:, :, c], psf, balance))
    return np.stack(out_channels, axis=2)


def richardson_lucy_channel(channel: np.ndarray, psf: np.ndarray, iterations: int = 20) -> np.ndarray:
    eps = 1e-8
    estimate = np.full_like(channel, 0.5)
    psf_mirror = np.flipud(np.fliplr(psf))

    for _ in range(iterations):
        conv_est = fft_convolve2d(estimate, psf)
        relative_blur = channel / (conv_est + eps)
        estimate *= fft_convolve2d(relative_blur, psf_mirror)
        estimate = np.clip(estimate, 0.0, 1.0)

    return estimate


def richardson_lucy(image: np.ndarray, psf: np.ndarray, iterations: int = 20) -> np.ndarray:
    if image.ndim == 2:
        return richardson_lucy_channel(image, psf, iterations)

    out_channels = []
    for c in range(image.shape[2]):
        out_channels.append(richardson_lucy_channel(image[:, :, c], psf, iterations))
    return np.stack(out_channels, axis=2)


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.where(mask > 0, image, 0.0).astype(np.float32)
    return (image * (mask[..., None] > 0)).astype(np.float32)


def compute_region_bbox(
    region_mask: np.ndarray,
    padding: int,
    min_weight: float = 1e-3,
) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(region_mask > min_weight)
    if xx.size == 0 or yy.size == 0:
        return None

    y0 = max(0, int(yy.min()) - padding)
    y1 = min(region_mask.shape[0], int(yy.max()) + padding + 1)
    x0 = max(0, int(xx.min()) - padding)
    x1 = min(region_mask.shape[1], int(xx.max()) + padding + 1)
    return y0, y1, x0, x1


def spatially_varying_deconvolution(
    image: np.ndarray,
    psfs: dict[str, np.ndarray],
    mask: np.ndarray,
    anchors: dict[str, tuple[float, float]],
    method: str,
    wiener_balance: float,
    rl_iters: int,
    center_radius: float,
    feather_sigma: float,
) -> np.ndarray:
    region_masks = compute_local_region_masks(
        mask,
        anchors,
        center_radius=center_radius,
        feather_sigma=feather_sigma,
    )
    accum = np.zeros_like(image, dtype=np.float32)

    for name, psf in psfs.items():
        padding = max(psf.shape) + int(np.ceil(3 * feather_sigma))
        bbox = compute_region_bbox(region_masks[name], padding=padding)
        if bbox is None:
            continue

        y0, y1, x0, x1 = bbox
        image_crop = image[y0:y1, x0:x1]
        weight = region_masks[name][y0:y1, x0:x1]

        if method == "wiener":
            restored = wiener_deconvolution(image_crop, psf, balance=wiener_balance)
        else:
            restored = richardson_lucy(image_crop, psf, iterations=rl_iters)

        if restored.ndim == 2:
            accum[y0:y1, x0:x1] += restored * weight
        else:
            accum[y0:y1, x0:x1] += restored * weight[..., None]

    return apply_mask_to_image(np.clip(accum, 0.0, 1.0), mask)


def draw_anchor_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    anchors: dict[str, tuple[float, float]],
    output_path: Path,
) -> None:
    base = np.clip(to_grayscale_float(image) * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    vis[mask == 0] = (0, 0, 0)

    colors = {
        "centre": (0, 255, 255),
        "haut": (255, 0, 0),
        "bas": (0, 255, 0),
        "gauche": (0, 128, 255),
        "droite": (255, 128, 0),
    }
    for name, (x, y) in anchors.items():
        cv2.circle(vis, (int(round(x)), int(round(y))), 12, colors[name], -1)
        cv2.putText(
            vis,
            name,
            (int(round(x)) + 10, int(round(y)) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colors[name],
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), vis)


def save_region_map(
    mask: np.ndarray,
    region_masks: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colors = {
        "centre": np.array([0, 255, 255], dtype=np.float32),
        "haut": np.array([255, 0, 0], dtype=np.float32),
        "bas": np.array([0, 255, 0], dtype=np.float32),
        "gauche": np.array([0, 128, 255], dtype=np.float32),
        "droite": np.array([255, 128, 0], dtype=np.float32),
    }
    accum = np.zeros_like(vis, dtype=np.float32)
    for name, region in region_masks.items():
        accum += region[..., None] * colors[name]
    vis = np.clip(accum, 0, 255).astype(np.uint8)
    vis[mask == 0] = 0
    cv2.imwrite(str(output_path), vis)


def make_comparison_strip(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    vis = []
    max_height = 0
    for img, label in zip(images, labels):
        if img.ndim == 2:
            bgr = cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            bgr = np.clip(img * 255, 0, 255).astype(np.uint8)

        cv2.putText(
            bgr,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        vis.append(bgr)
        max_height = max(max_height, bgr.shape[0])

    resized = []
    for bgr in vis:
        if bgr.shape[0] == max_height:
            resized.append(bgr)
            continue
        scale = max_height / bgr.shape[0]
        target_width = max(1, int(round(bgr.shape[1] * scale)))
        resized.append(cv2.resize(bgr, (target_width, max_height), interpolation=cv2.INTER_AREA))

    return cv2.hconcat(resized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Utilise une PSF unique ou plusieurs PSF locales pour deconvoluer une image "
            "avec Wiener / Richardson-Lucy."
        )
    )
    parser.add_argument("--input", required=True, help="Chemin image de reference")
    parser.add_argument("--psf", default=None, help="Chemin PSF unique (.npy ou image)")
    parser.add_argument("--aberrated-input", default=None, help="Image deja aberrée (optionnel)")
    parser.add_argument("--simulate-aberration", action="store_true", help="Convolue --input avec la PSF")
    parser.add_argument("--method", choices=["wiener", "rl", "both"], default="both")
    parser.add_argument("--wiener-balance", type=float, default=1e-3)
    parser.add_argument("--rl-iters", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs_deconvolution")
    parser.add_argument("--local-psf-centre", default=None, help="Image PSF locale du centre")
    parser.add_argument("--local-psf-haut", default=None, help="Image PSF locale du haut")
    parser.add_argument("--local-psf-bas", default=None, help="Image PSF locale du bas")
    parser.add_argument("--local-psf-gauche", default=None, help="Image PSF locale de gauche")
    parser.add_argument("--local-psf-droite", default=None, help="Image PSF locale de droite")
    parser.add_argument(
        "--manual-select-local",
        action="append",
        default=[],
        help="Nom d'une PSF locale a choisir manuellement dans {centre,haut,bas,gauche,droite}",
    )
    parser.add_argument("--local-psf-kernel-size", type=int, default=31)
    parser.add_argument("--local-psf-gaussian-sigma", type=float, default=0.0)
    parser.add_argument("--local-psf-downscale-factor", type=float, default=1.0)
    parser.add_argument("--mask-threshold", type=int, default=11)
    parser.add_argument("--mask-close-radius", type=int, default=9)
    parser.add_argument("--mask-margin-ratio", type=float, default=1.02)
    parser.add_argument(
        "--local-center-radius",
        type=float,
        default=0.35,
        help="Rayon normalise de la zone centrale dans l'ovale",
    )
    parser.add_argument(
        "--local-feather-sigma",
        type=float,
        default=9.0,
        help="Lissage des transitions entre zones locales en pixels",
    )
    return parser


def build_manual_args() -> argparse.Namespace:
    if not INPUT_IMAGE:
        raise ValueError(
            "Renseignez INPUT_IMAGE dans la section IMPLEMENTATION ou lancez le script avec --input."
        )

    return argparse.Namespace(
        input=INPUT_IMAGE,
        psf=PSF_PATH,
        aberrated_input=ABERRATED_INPUT,
        simulate_aberration=SIMULATE_ABERRATION,
        method=METHOD,
        wiener_balance=WIENER_BALANCE,
        rl_iters=RL_ITERS,
        output_dir=OUTPUT_DIR,
        local_psf_centre=LOCAL_PSF_CENTRE,
        local_psf_haut=LOCAL_PSF_HAUT,
        local_psf_bas=LOCAL_PSF_BAS,
        local_psf_gauche=LOCAL_PSF_GAUCHE,
        local_psf_droite=LOCAL_PSF_DROITE,
        manual_select_local=MANUAL_SELECT_LOCAL,
        local_psf_kernel_size=LOCAL_PSF_KERNEL_SIZE,
        local_psf_gaussian_sigma=LOCAL_PSF_GAUSSIAN_SIGMA,
        local_psf_downscale_factor=LOCAL_PSF_DOWNSCALE_FACTOR,
        mask_threshold=MASK_THRESHOLD,
        mask_close_radius=MASK_CLOSE_RADIUS,
        mask_margin_ratio=MASK_MARGIN_RATIO,
        local_center_radius=LOCAL_CENTER_RADIUS,
        local_feather_sigma=LOCAL_FEATHER_SIGMA,
    )


def main(args: argparse.Namespace) -> None:

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sharp = load_image(input_path)

    local_psf_paths = {
        "centre": args.local_psf_centre,
        "haut": args.local_psf_haut,
        "bas": args.local_psf_bas,
        "gauche": args.local_psf_gauche,
        "droite": args.local_psf_droite,
    }
    use_local_psfs = all(path is not None for path in local_psf_paths.values())
    manual_local = parse_local_psf_names(args.manual_select_local)

    if args.aberrated_input is not None:
        aberrated = load_image(Path(args.aberrated_input))
    elif args.simulate_aberration and args.psf is not None:
        psf = load_psf(Path(args.psf))
        aberrated = apply_aberration_with_psf(sharp, psf)
        save_image(out_dir / "aberrated_simulated.png", aberrated)
    else:
        raise ValueError("Utilisez --aberrated-input ou --simulate-aberration avec --psf.")

    results = []
    labels = []

    if use_local_psfs:
        local_psfs: dict[str, np.ndarray] = {}
        for name, raw_path in local_psf_paths.items():
            center = None
            if name in manual_local:
                image = load_grayscale_image(Path(raw_path))
                center = choose_psf_center_interactively(image)
            psf, _, detected_center = estimate_psf_from_image_file(
                Path(raw_path),
                args.local_psf_kernel_size,
                center=center,
                gaussian_sigma=args.local_psf_gaussian_sigma,
                downscale_factor=args.local_psf_downscale_factor,
            )
            local_psfs[name] = psf
            np.save(out_dir / f"estimated_psf_{name}.npy", psf)
            print(f"PSF {name}: centre detecte x={detected_center[0]}, y={detected_center[1]}")

        mask, ellipse_center, ellipse_axes, ellipse_angle = build_useful_region_mask(
            aberrated,
            threshold=args.mask_threshold,
            close_radius=args.mask_close_radius,
            margin_ratio=args.mask_margin_ratio,
        )
        anchors = compute_local_anchor_points(mask)
        region_masks = compute_local_region_masks(
            mask,
            anchors,
            center_radius=args.local_center_radius,
            feather_sigma=args.local_feather_sigma,
        )
        cv2.imwrite(str(out_dir / "useful_region_mask.png"), mask)
        draw_anchor_visualization(aberrated, mask, anchors, out_dir / "local_psf_anchor_map.png")
        save_region_map(mask, region_masks, out_dir / "local_psf_region_map.png")

        if args.method in ("wiener", "both"):
            restored_wiener = spatially_varying_deconvolution(
                aberrated,
                local_psfs,
                mask,
                anchors,
                method="wiener",
                wiener_balance=args.wiener_balance,
                rl_iters=args.rl_iters,
                center_radius=args.local_center_radius,
                feather_sigma=args.local_feather_sigma,
            )
            save_image(out_dir / "restored_wiener_local.png", restored_wiener)
            results.append(restored_wiener)
            labels.append(f"Wiener local k={args.wiener_balance}")

        if args.method in ("rl", "both"):
            restored_rl = spatially_varying_deconvolution(
                aberrated,
                local_psfs,
                mask,
                anchors,
                method="rl",
                wiener_balance=args.wiener_balance,
                rl_iters=args.rl_iters,
                center_radius=args.local_center_radius,
                feather_sigma=args.local_feather_sigma,
            )
            save_image(out_dir / "restored_rl_local.png", restored_rl)
            results.append(restored_rl)
            labels.append(f"RL local it={args.rl_iters}")
    else:
        if args.psf is None:
            raise ValueError("Fournissez --psf pour une PSF unique, ou les 5 arguments --local-psf-*.")

        psf = load_psf(Path(args.psf))

        if args.method in ("wiener", "both"):
            restored_wiener = wiener_deconvolution(aberrated, psf, balance=args.wiener_balance)
            save_image(out_dir / "restored_wiener.png", restored_wiener)
            results.append(restored_wiener)
            labels.append(f"Wiener k={args.wiener_balance}")

        if args.method in ("rl", "both"):
            restored_rl = richardson_lucy(aberrated, psf, iterations=args.rl_iters)
            save_image(out_dir / "restored_rl.png", restored_rl)
            results.append(restored_rl)
            labels.append(f"Richardson-Lucy it={args.rl_iters}")

    comp_images = [sharp, aberrated] + results
    comp_labels = ["Original", "Aberree"] + labels
    strip = make_comparison_strip(comp_images, comp_labels)
    cv2.imwrite(str(out_dir / "comparison_strip.png"), strip)

    print("Termine.")
    print(f"Dossier resultats: {out_dir.resolve()}")


# -------------------------------------------------------------------------
# ------------------------- IMPLEMENTATION  -------------------------------
# -------------------------------------------------------------------------

# Remplir cette section si vous voulez lancer le script directement depuis
# l'editeur sans arguments en ligne de commande.

INPUT_IMAGE = ""
OUTPUT_DIR = "outputs_deconvolution"

# Mode d'entree :
# - soit renseigner ABERRATED_INPUT avec une image deja floutee
# - soit mettre SIMULATE_ABERRATION = True et fournir PSF_PATH
PSF_PATH = None
ABERRATED_INPUT = None
SIMULATE_ABERRATION = False

METHOD = "both"
WIENER_BALANCE = 1e-3
RL_ITERS = 20

# Laisser les 5 chemins a None pour utiliser une PSF unique via PSF_PATH.
# Renseigner les 5 pour activer la deconvolution locale a PSF multiples.
LOCAL_PSF_CENTRE = None
LOCAL_PSF_HAUT = None
LOCAL_PSF_BAS = None
LOCAL_PSF_GAUCHE = None
LOCAL_PSF_DROITE = None

# Exemple: ["centre", "haut"]
MANUAL_SELECT_LOCAL = []
LOCAL_PSF_KERNEL_SIZE = 31
LOCAL_PSF_GAUSSIAN_SIGMA = 0.0
LOCAL_PSF_DOWNSCALE_FACTOR = 1.0

MASK_THRESHOLD = 11
MASK_CLOSE_RADIUS = 9
MASK_MARGIN_RATIO = 1.02
LOCAL_CENTER_RADIUS = 0.35
LOCAL_FEATHER_SIGMA = 9.0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(build_parser().parse_args())
    else:
        main(build_manual_args())
