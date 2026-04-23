import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from deconvolution import (
    apply_mask_to_image,
    build_useful_region_mask,
    load_image,
    richardson_lucy,
    save_image,
    to_grayscale_float,
    wiener_deconvolution,
)


def estimate_tile_blur_sigma(
    gray_tile: np.ndarray,
    mask_tile: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    eps: float = 1e-6,
) -> float:
    valid = mask_tile > 0
    if np.count_nonzero(valid) < 64:
        return sigma_max

    gray_u8 = np.clip(gray_tile * 255.0, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    lap_var = float(np.var(lap[valid]))
    grad_p90 = float(np.percentile(grad[valid], 90))

    # Score empirique: plus il est eleve, plus la tuile est nette.
    sharpness = lap_var + 0.25 * grad_p90
    blur_ratio = 1.0 / np.sqrt(sharpness + eps)

    # Echelle empirique stable pour convertir le score en sigma gaussien local.
    sigma = sigma_min + (sigma_max - sigma_min) * np.clip(blur_ratio / 0.35, 0.0, 1.0)
    return float(np.clip(sigma, sigma_min, sigma_max))


def gaussian_psf(kernel_size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    psf = np.maximum(psf, 0.0)
    psf /= float(psf.sum())
    return psf.astype(np.float32)


def iter_tile_boxes(mask: np.ndarray, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    yy, xx = np.nonzero(mask > 0)
    if xx.size == 0 or yy.size == 0:
        raise ValueError("Le masque de la zone utile est vide.")

    y_min, y_max = int(yy.min()), int(yy.max()) + 1
    x_min, x_max = int(xx.min()), int(xx.max()) + 1

    step = max(8, tile_size - overlap)
    boxes: list[tuple[int, int, int, int]] = []

    for y0 in range(y_min, y_max, step):
        for x0 in range(x_min, x_max, step):
            y1 = min(mask.shape[0], y0 + tile_size)
            x1 = min(mask.shape[1], x0 + tile_size)
            if np.count_nonzero(mask[y0:y1, x0:x1]) == 0:
                continue
            boxes.append((y0, y1, x0, x1))

    return boxes


def feather_window(height: int, width: int) -> np.ndarray:
    wy = np.hanning(height) if height > 2 else np.ones(height, dtype=np.float32)
    wx = np.hanning(width) if width > 2 else np.ones(width, dtype=np.float32)
    window = np.outer(wy, wx).astype(np.float32)
    return np.maximum(window, 1e-3)


def save_sigma_map(
    sigma_map: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    sigma_min: float,
    sigma_max: float,
) -> None:
    normalized = (sigma_map - sigma_min) / max(1e-6, sigma_max - sigma_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    gray = (normalized * 255.0).astype(np.uint8)
    gray[mask == 0] = 0
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[mask == 0] = 0
    cv2.imwrite(str(output_path), color)


def tile_adaptive_deconvolution(
    image: np.ndarray,
    mask: np.ndarray,
    method: str,
    tile_size: int,
    overlap: int,
    kernel_size: int,
    sigma_min: float,
    sigma_max: float,
    wiener_balance: float,
    rl_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size doit etre impair et >= 3.")

    gray = to_grayscale_float(image)
    accum = np.zeros_like(image, dtype=np.float32)
    weight_sum = np.zeros(gray.shape, dtype=np.float32)
    sigma_sum = np.zeros(gray.shape, dtype=np.float32)
    sigma_weight = np.zeros(gray.shape, dtype=np.float32)

    for y0, y1, x0, x1 in iter_tile_boxes(mask, tile_size, overlap):
        tile_mask = mask[y0:y1, x0:x1]
        tile_gray = gray[y0:y1, x0:x1]
        sigma = estimate_tile_blur_sigma(tile_gray, tile_mask, sigma_min, sigma_max)
        psf = gaussian_psf(kernel_size, sigma)

        crop = image[y0:y1, x0:x1]
        if method == "wiener":
            restored = wiener_deconvolution(crop, psf, balance=wiener_balance)
        else:
            restored = richardson_lucy(crop, psf, iterations=rl_iters)

        local_weight = feather_window(y1 - y0, x1 - x0) * (tile_mask > 0)
        if restored.ndim == 2:
            accum[y0:y1, x0:x1] += restored * local_weight
        else:
            accum[y0:y1, x0:x1] += restored * local_weight[..., None]

        weight_sum[y0:y1, x0:x1] += local_weight
        sigma_sum[y0:y1, x0:x1] += sigma * local_weight
        sigma_weight[y0:y1, x0:x1] += local_weight

    safe_weight = np.maximum(weight_sum, 1e-6)
    if accum.ndim == 2:
        restored = accum / safe_weight
    else:
        restored = accum / safe_weight[..., None]

    sigma_map = sigma_sum / np.maximum(sigma_weight, 1e-6)
    restored = apply_mask_to_image(np.clip(restored, 0.0, 1.0), mask)
    sigma_map[mask == 0] = 0.0
    return restored, sigma_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estime un flou local par tuiles directement depuis l'image, "
            "puis applique une deconvolution locale avec une PSF gaussienne par tuile."
        )
    )
    parser.add_argument("--input", required=True, help="Image a deconvoluer")
    parser.add_argument("--output-dir", default="outputs_tile_deconvolution")
    parser.add_argument("--method", choices=["wiener", "rl"], default="wiener")
    parser.add_argument("--tile-size", type=int, default=192, help="Taille des tuiles en pixels")
    parser.add_argument("--tile-overlap", type=int, default=64, help="Recouvrement entre tuiles en pixels")
    parser.add_argument("--kernel-size", type=int, default=21, help="Taille de la PSF gaussienne locale")
    parser.add_argument("--sigma-min", type=float, default=0.8, help="Sigma minimal de la PSF locale")
    parser.add_argument("--sigma-max", type=float, default=3.5, help="Sigma maximal de la PSF locale")
    parser.add_argument("--wiener-balance", type=float, default=1e-2)
    parser.add_argument("--rl-iters", type=int, default=8)
    parser.add_argument("--mask-threshold", type=int, default=11)
    parser.add_argument("--mask-close-radius", type=int, default=9)
    parser.add_argument("--mask-margin-ratio", type=float, default=1.02)
    return parser


def build_manual_args() -> argparse.Namespace:
    if not INPUT_IMAGE:
        raise ValueError(
            "Renseignez INPUT_IMAGE dans la section IMPLEMENTATION ou lancez le script avec --input."
        )

    return argparse.Namespace(
        input=INPUT_IMAGE,
        output_dir=OUTPUT_DIR,
        method=METHOD,
        tile_size=TILE_SIZE,
        tile_overlap=TILE_OVERLAP,
        kernel_size=KERNEL_SIZE,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        wiener_balance=WIENER_BALANCE,
        rl_iters=RL_ITERS,
        mask_threshold=MASK_THRESHOLD,
        mask_close_radius=MASK_CLOSE_RADIUS,
        mask_margin_ratio=MASK_MARGIN_RATIO,
    )


def main(args: argparse.Namespace) -> None:

    image = load_image(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask, _, _, _ = build_useful_region_mask(
        image,
        threshold=args.mask_threshold,
        close_radius=args.mask_close_radius,
        margin_ratio=args.mask_margin_ratio,
    )
    cv2.imwrite(str(output_dir / "useful_region_mask.png"), mask)

    restored, sigma_map = tile_adaptive_deconvolution(
        image=image,
        mask=mask,
        method=args.method,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
        kernel_size=args.kernel_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        wiener_balance=args.wiener_balance,
        rl_iters=args.rl_iters,
    )

    restored_name = "restored_tile_wiener.png" if args.method == "wiener" else "restored_tile_rl.png"
    save_image(output_dir / restored_name, restored)
    save_sigma_map(
        sigma_map,
        mask,
        output_dir / "tile_sigma_map.png",
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )

    print("Termine.")
    print(f"Resultats: {output_dir.resolve()}")


# -------------------------------------------------------------------------
# ------------------------- IMPLEMENTATION  -------------------------------
# -------------------------------------------------------------------------

# Remplir cette section si vous voulez lancer le script directement depuis
# l'editeur sans arguments en ligne de commande.

INPUT_IMAGE = "Deconvolution\Image_examples\irtest.jpg"
OUTPUT_DIR = "Deconvolution\outputs_tile_local_deconvolution"

METHOD = "wiener"
TILE_SIZE = 192
TILE_OVERLAP = 64
KERNEL_SIZE = 21
SIGMA_MIN = 0.8
SIGMA_MAX = 3.5
WIENER_BALANCE = 1e-2
RL_ITERS = 8

MASK_THRESHOLD = 11
MASK_CLOSE_RADIUS = 9
MASK_MARGIN_RATIO = 1.02


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(build_parser().parse_args())
    else:
        main(build_manual_args())
