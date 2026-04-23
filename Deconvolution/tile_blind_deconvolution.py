import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from deconvolution import (
    apply_mask_to_image,
    build_useful_region_mask,
    fft_convolve2d,
    load_image,
    save_image,
    to_grayscale_float,
)
from tile_local_deconvolution import feather_window


def normalize_psf(psf: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    psf = np.maximum(psf, 0.0)
    total = float(psf.sum())
    if total <= eps:
        psf = np.zeros_like(psf, dtype=np.float32)
        cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
        psf[cy, cx] = 1.0
        return psf
    return (psf / total).astype(np.float32)


def gaussian_psf(kernel_size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    return normalize_psf(psf)


def kernel_fft(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.fft.fft2(np.fft.ifftshift(kernel), s=shape)


def fft_convolve2d_cached(image: np.ndarray, kernel_fft_value: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.fft2(image) * kernel_fft_value).real


def richardson_lucy_latent(
    observed: np.ndarray,
    psf: np.ndarray,
    iterations: int,
    use_cached_fft: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    estimate = np.clip(observed.copy(), eps, 1.0).astype(np.float32)
    psf_mirror = np.flipud(np.fliplr(psf))
    psf_fft = None
    psf_mirror_fft = None

    if use_cached_fft:
        psf_fft = kernel_fft(psf, estimate.shape)
        psf_mirror_fft = kernel_fft(psf_mirror, estimate.shape)

    for _ in range(iterations):
        if use_cached_fft:
            conv_est = fft_convolve2d_cached(estimate, psf_fft)
        else:
            conv_est = fft_convolve2d(estimate, psf)
        relative_blur = observed / (conv_est + eps)
        if use_cached_fft:
            estimate *= fft_convolve2d_cached(relative_blur, psf_mirror_fft)
        else:
            estimate *= fft_convolve2d(relative_blur, psf_mirror)
        estimate = np.clip(estimate, 0.0, 1.0)

    return estimate


def update_psf_from_latent(
    observed: np.ndarray,
    latent: np.ndarray,
    psf: np.ndarray,
    psf_smooth_sigma: float,
    use_cached_fft: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    if use_cached_fft:
        psf_fft = kernel_fft(psf, latent.shape)
        relative_blur = observed / (fft_convolve2d_cached(latent, psf_fft) + eps)
    else:
        relative_blur = observed / (fft_convolve2d(latent, psf) + eps)
    latent_mirror = np.flipud(np.fliplr(latent))
    full_update = fft_convolve2d(relative_blur, latent_mirror)
    cy, cx = full_update.shape[0] // 2, full_update.shape[1] // 2
    kh, kw = psf.shape
    ky, kx = kh // 2, kw // 2

    # Near image borders, the last tile can be smaller than the requested kernel.
    # Extract the centered support and pad it back to the PSF size if needed.
    y0 = max(0, cy - ky)
    y1 = min(full_update.shape[0], cy + ky + 1)
    x0 = max(0, cx - kx)
    x1 = min(full_update.shape[1], cx + kx + 1)
    psf_support = full_update[y0:y1, x0:x1]

    if psf_support.shape != psf.shape:
        padded = np.zeros_like(psf, dtype=np.float32)
        py = (kh - psf_support.shape[0]) // 2
        px = (kw - psf_support.shape[1]) // 2
        padded[py : py + psf_support.shape[0], px : px + psf_support.shape[1]] = psf_support
        psf_support = padded

    psf_update = psf * psf_support
    psf_update = normalize_psf(psf_update, eps=eps)

    if psf_smooth_sigma > 0:
        psf_update = cv2.GaussianBlur(psf_update, (0, 0), psf_smooth_sigma)
        psf_update = normalize_psf(psf_update, eps=eps)

    return psf_update


def blind_deconvolution_tile(
    observed: np.ndarray,
    kernel_size: int,
    init_sigma: float,
    blind_iters: int,
    latent_iters: int,
    psf_smooth_sigma: float,
    use_cached_fft: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    psf = gaussian_psf(kernel_size, init_sigma)
    latent = np.clip(observed.copy(), 0.0, 1.0).astype(np.float32)

    for _ in range(blind_iters):
        latent = richardson_lucy_latent(
            observed,
            psf,
            iterations=latent_iters,
            use_cached_fft=use_cached_fft,
        )
        psf = update_psf_from_latent(
            observed=observed,
            latent=latent,
            psf=psf,
            psf_smooth_sigma=psf_smooth_sigma,
            use_cached_fft=use_cached_fft,
        )

    return latent, psf


def process_tile(
    gray: np.ndarray,
    mask: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    kernel_size: int,
    init_sigma: float,
    blind_iters: int,
    latent_iters: int,
    psf_smooth_sigma: float,
    use_cached_fft: bool,
) -> tuple[int, int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    tile_mask = mask[y0:y1, x0:x1]
    observed = gray[y0:y1, x0:x1]
    latent, psf = blind_deconvolution_tile(
        observed=observed,
        kernel_size=kernel_size,
        init_sigma=init_sigma,
        blind_iters=blind_iters,
        latent_iters=latent_iters,
        psf_smooth_sigma=psf_smooth_sigma,
        use_cached_fft=use_cached_fft,
    )
    local_weight = feather_window(y1 - y0, x1 - x0) * (tile_mask > 0)
    return y0, y1, x0, x1, latent, psf, local_weight


def psf_to_intensity_map(
    psf_grid: list[list[np.ndarray | None]],
    grid_shape: tuple[int, int],
    tile_size: int,
) -> np.ndarray:
    rows, cols = grid_shape
    panel = np.zeros((rows * tile_size, cols * tile_size), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            psf = psf_grid[r][c]
            if psf is None:
                continue
            resized = cv2.resize(psf, (tile_size, tile_size), interpolation=cv2.INTER_CUBIC)
            resized = resized / max(1e-8, float(resized.max()))
            panel[r * tile_size : (r + 1) * tile_size, c * tile_size : (c + 1) * tile_size] = resized

    return np.clip(panel * 255.0, 0, 255).astype(np.uint8)


def tile_blind_deconvolution(
    image: np.ndarray,
    mask: np.ndarray,
    tile_size: int,
    overlap: int,
    kernel_size: int,
    init_sigma: float,
    blind_iters: int,
    latent_iters: int,
    psf_smooth_sigma: float,
    workers: int = 1,
    use_cached_fft: bool = False,
) -> tuple[np.ndarray, list[list[np.ndarray | None]], tuple[int, int]]:
    gray = to_grayscale_float(image)
    accum = np.zeros_like(gray, dtype=np.float32)
    weight_sum = np.zeros_like(gray, dtype=np.float32)

    yy, xx = np.nonzero(mask > 0)
    y_min, y_max = int(yy.min()), int(yy.max()) + 1
    x_min, x_max = int(xx.min()), int(xx.max()) + 1
    step = max(8, tile_size - overlap)
    rows = max(1, int(np.ceil((y_max - y_min) / step)))
    cols = max(1, int(np.ceil((x_max - x_min) / step)))
    psf_grid: list[list[np.ndarray | None]] = [[None for _ in range(cols)] for _ in range(rows)]

    tile_jobs: list[tuple[int, int, int, int, int, int]] = []
    for row, y0_base in enumerate(range(y_min, y_max, step)):
        for col, x0_base in enumerate(range(x_min, x_max, step)):
            y0 = y0_base
            x0 = x0_base
            y1 = min(mask.shape[0], y0 + tile_size)
            x1 = min(mask.shape[1], x0 + tile_size)

            tile_mask = mask[y0:y1, x0:x1]
            if np.count_nonzero(tile_mask) == 0:
                continue

            tile_jobs.append((row, col, y0, y1, x0, x1))

    if workers <= 1:
        tile_results = [
            (
                row,
                col,
                *process_tile(
                    gray,
                    mask,
                    y0,
                    y1,
                    x0,
                    x1,
                    kernel_size,
                    init_sigma,
                    blind_iters,
                    latent_iters,
                    psf_smooth_sigma,
                    use_cached_fft,
                ),
            )
            for row, col, y0, y1, x0, x1 in tile_jobs
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                (
                    row,
                    col,
                    executor.submit(
                        process_tile,
                        gray,
                        mask,
                        y0,
                        y1,
                        x0,
                        x1,
                        kernel_size,
                        init_sigma,
                        blind_iters,
                        latent_iters,
                        psf_smooth_sigma,
                        use_cached_fft,
                    ),
                )
                for row, col, y0, y1, x0, x1 in tile_jobs
            ]
            tile_results = [
                (row, col, *future.result())
                for row, col, future in futures
            ]

    tile_index = 0
    for row, col, y0, y1, x0, x1, latent, psf, local_weight in tile_results:
            accum[y0:y1, x0:x1] += latent * local_weight
            weight_sum[y0:y1, x0:x1] += local_weight
            psf_grid[row][col] = psf
            tile_index += 1

    restored = accum / np.maximum(weight_sum, 1e-6)
    restored = apply_mask_to_image(np.clip(restored, 0.0, 1.0), mask)
    return restored, psf_grid, (rows, cols)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Blind deconvolution locale par tuiles: estime une PSF par tuile directement "
            "depuis l'image, puis fusionne les tuiles restaurées."
        )
    )
    parser.add_argument("--input", required=True, help="Image a deconvoluer")
    parser.add_argument("--output-dir", default="outputs_tile_blind_deconvolution")
    parser.add_argument("--tile-size", type=int, default=160, help="Taille des tuiles en pixels")
    parser.add_argument("--tile-overlap", type=int, default=64, help="Recouvrement entre tuiles")
    parser.add_argument("--kernel-size", type=int, default=17, help="Taille impaire de la PSF estimee")
    parser.add_argument("--init-sigma", type=float, default=1.8, help="Sigma initial de la PSF")
    parser.add_argument("--blind-iters", type=int, default=4, help="Iterations d'estimation alternee")
    parser.add_argument("--latent-iters", type=int, default=6, help="Iterations RL pour l'image latente")
    parser.add_argument("--psf-smooth-sigma", type=float, default=0.8, help="Lissage de la PSF entre iterations")
    parser.add_argument("--workers", type=int, default=1, help="Nombre de tuiles traitees en parallele")
    parser.add_argument(
        "--cache-kernel-fft",
        action="store_true",
        help="Reutilise la FFT de la PSF et de la PSF miroir dans les boucles iteratives",
    )
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
        tile_size=TILE_SIZE,
        tile_overlap=TILE_OVERLAP,
        kernel_size=KERNEL_SIZE,
        init_sigma=INIT_SIGMA,
        blind_iters=BLIND_ITERS,
        latent_iters=LATENT_ITERS,
        psf_smooth_sigma=PSF_SMOOTH_SIGMA,
        workers=WORKERS,
        cache_kernel_fft=CACHE_KERNEL_FFT,
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

    restored, psf_grid, grid_shape = tile_blind_deconvolution(
        image=image,
        mask=mask,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
        kernel_size=args.kernel_size,
        init_sigma=args.init_sigma,
        blind_iters=args.blind_iters,
        latent_iters=args.latent_iters,
        psf_smooth_sigma=args.psf_smooth_sigma,
        workers=max(1, args.workers),
        use_cached_fft=args.cache_kernel_fft,
    )

    save_image(output_dir / "restored_tile_blind.png", restored)
    psf_panel = psf_to_intensity_map(psf_grid, grid_shape, tile_size=args.kernel_size * 8)
    cv2.imwrite(str(output_dir / "tile_psf_grid.png"), psf_panel)

    print("Termine.")
    print(f"Resultats: {output_dir.resolve()}")


# -------------------------------------------------------------------------
# ------------------------- IMPLEMENTATION  -------------------------------
# -------------------------------------------------------------------------

# Remplir cette section si vous voulez lancer le script directement depuis
# l'editeur sans arguments en ligne de commande.

INPUT_IMAGE = "Deconvolution\Image_examples\irtest.jpg"
OUTPUT_DIR = "Deconvolution\outputs_tile_blind_deconvolution"

TILE_SIZE = 160
TILE_OVERLAP = 64
KERNEL_SIZE = 17
INIT_SIGMA = 1.8
BLIND_ITERS = 4
LATENT_ITERS = 6
PSF_SMOOTH_SIGMA = 0.8
WORKERS = 1
CACHE_KERNEL_FFT = False

MASK_THRESHOLD = 11
MASK_CLOSE_RADIUS = 9
MASK_MARGIN_RATIO = 1.02


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(build_parser().parse_args())
    else:
        main(build_manual_args())
