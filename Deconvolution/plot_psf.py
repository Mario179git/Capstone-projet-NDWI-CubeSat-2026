import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_grayscale_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image: {path}")
    return image.astype(np.float32)


def extract_psf_patch(
    image: np.ndarray,
    kernel_size: int,
    center: tuple[int, int] | None = None,
    gaussian_sigma: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int]]:
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size doit etre impair et >= 3.")

    half = kernel_size // 2

    if center is None:
        y, x = np.unravel_index(np.argmax(image), image.shape)
    else:
        x, y = center

    y = int(np.clip(y, half, image.shape[0] - half - 1))
    x = int(np.clip(x, half, image.shape[1] - half - 1))

    patch = image[y - half : y + half + 1, x - half : x + half + 1].copy()
    patch -= patch.min()

    if gaussian_sigma > 0:
        patch = cv2.GaussianBlur(patch, (0, 0), gaussian_sigma)
        patch = np.maximum(patch, 0.0)

    total = float(patch.sum())
    if total <= 0:
        raise ValueError("Impossible d'estimer une PSF valide a partir de cette zone.")

    patch /= total
    return patch, (x, y)


def estimate_psf_from_image_file(
    image_path: Path,
    kernel_size: int,
    center: tuple[int, int] | None = None,
    gaussian_sigma: float = 0.0,
    downscale_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    image = load_grayscale_image(image_path)
    psf, detected_center = extract_psf_patch(
        image,
        kernel_size,
        center=center,
        gaussian_sigma=gaussian_sigma,
    )
    psf = resize_psf(psf, downscale_factor)
    return psf, image, detected_center


def resize_psf(psf: np.ndarray, scale_factor: float) -> np.ndarray:
    if scale_factor <= 0:
        raise ValueError("scale_factor doit etre > 0.")

    if scale_factor == 1.0:
        return psf

    target_h = max(3, int(round(psf.shape[0] / scale_factor)))
    target_w = max(3, int(round(psf.shape[1] / scale_factor)))

    if target_h % 2 == 0:
        target_h += 1
    if target_w % 2 == 0:
        target_w += 1

    resized = cv2.resize(psf, (target_w, target_h), interpolation=cv2.INTER_AREA)
    resized = np.maximum(resized, 0.0)

    total = float(resized.sum())
    if total <= 0:
        raise ValueError("Impossible de renormaliser la PSF apres reduction.")

    return resized / total


def choose_psf_center_interactively(image: np.ndarray) -> tuple[int, int]:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image, cmap="gray")
    ax.set_title("Cliquez sur le centre de la PSF, puis fermez la fenetre si necessaire")
    ax.axis("off")

    points = plt.ginput(1, timeout=-1)
    plt.close(fig)

    if not points:
        raise RuntimeError("Aucun point selectionne pour la PSF.")

    x, y = points[0]
    return int(round(x)), int(round(y))


def plot_psf(
    psf: np.ndarray,
    image: np.ndarray,
    center: tuple[int, int],
    output_dir: Path,
    show: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y = center
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5))

    axes[0].imshow(image, cmap="gray")
    axes[0].scatter([x], [y], c="red", s=40)
    axes[0].set_title("Image et centre PSF")
    axes[0].axis("off")

    heatmap = axes[1].imshow(psf, cmap="hot")
    axes[1].set_title("PSF estimee")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
    axes[2].plot(psf[cy, :], label="Coupe horizontale")
    axes[2].plot(psf[:, cx], label="Coupe verticale")
    axes[2].set_title("Profils de la PSF")
    axes[2].set_xlabel("Pixels")
    axes[2].set_ylabel("Amplitude normalisee")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "psf_plot.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_local_psf_specs(specs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Format invalide pour --local-psf: {spec}. Utilisez nom=chemin")
        name, raw_path = spec.split("=", 1)
        name = name.strip().lower()
        if name not in {"centre", "haut", "bas", "gauche", "droite"}:
            raise ValueError(f"Nom de PSF locale invalide: {name}")
        parsed[name] = Path(raw_path.strip())
    return parsed


def parse_local_psf_names(names: list[str]) -> set[str]:
    parsed: set[str] = set()
    for name in names:
        key = name.strip().lower()
        if key not in {"centre", "haut", "bas", "gauche", "droite"}:
            raise ValueError(f"Nom de PSF locale invalide: {name}")
        parsed.add(key)
    return parsed


def plot_local_psf_summary(
    results: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]],
    output_dir: Path,
    show: bool,
) -> None:
    order = ["centre", "haut", "bas", "gauche", "droite"]
    available = [name for name in order if name in results]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 2, figsize=(9, 4 * len(available)))
    if len(available) == 1:
        axes = np.array([axes])

    for row, name in enumerate(available):
        psf, image, center = results[name]
        x, y = center
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].scatter([x], [y], c="red", s=40)
        axes[row, 0].set_title(f"{name.capitalize()} - image PSF")
        axes[row, 0].axis("off")

        heatmap = axes[row, 1].imshow(psf, cmap="hot")
        axes[row, 1].set_title(f"{name.capitalize()} - PSF estimee")
        axes[row, 1].axis("off")
        fig.colorbar(heatmap, ax=axes[row, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_dir / "local_psf_summary.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estime une PSF locale a partir du point le plus lumineux d'une image "
            "et la trace."
        )
    )
    parser.add_argument("--input", default=None, help="Chemin vers l'image d'entree")
    parser.add_argument("--kernel-size", type=int, default=31, help="Taille impaire de la PSF estimee")
    parser.add_argument("--x", type=int, default=None, help="Abscisse du centre PSF (optionnel)")
    parser.add_argument("--y", type=int, default=None, help="Ordonnee du centre PSF (optionnel)")
    parser.add_argument("--output-dir", default="outputs_psf", help="Dossier de sortie")
    parser.add_argument("--save-npy", action="store_true", help="Sauvegarde la PSF dans un fichier .npy")
    parser.add_argument("--show", action="store_true", help="Affiche la figure matplotlib")
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.0,
        help="Sigma du flou gaussien applique a la PSF extraite avant normalisation",
    )
    parser.add_argument(
        "--manual-select",
        action="store_true",
        help="Permet de choisir manuellement le centre de la PSF en cliquant sur l'image",
    )
    parser.add_argument(
        "--downscale-factor",
        type=float,
        default=1.0,
        help="Reduit la taille de la PSF par ce facteur apres extraction (ex: 10 -> PSF 10x plus petite)",
    )
    parser.add_argument(
        "--local-psf",
        action="append",
        default=[],
        help="PSF locale au format nom=chemin, avec nom dans {centre,haut,bas,gauche,droite}",
    )
    parser.add_argument(
        "--manual-select-local",
        action="append",
        default=[],
        help="Nom d'une PSF locale a choisir manuellement dans {centre,haut,bas,gauche,droite}",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.local_psf:
        local_specs = parse_local_psf_specs(args.local_psf)
        manual_local = parse_local_psf_names(args.manual_select_local)
        results: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}

        for name, image_path in local_specs.items():
            center = None
            if name in manual_local:
                image = load_grayscale_image(image_path)
                center = choose_psf_center_interactively(image)
            psf, image, detected_center = estimate_psf_from_image_file(
                image_path,
                args.kernel_size,
                center=center,
                gaussian_sigma=args.gaussian_sigma,
                downscale_factor=args.downscale_factor,
            )
            results[name] = (psf, image, detected_center)
            np.save(output_dir / f"estimated_psf_{name}.npy", psf)
            plot_psf(psf, image, detected_center, output_dir / name, show=False)

        plot_local_psf_summary(results, output_dir, show=args.show)
        print("PSF locales estimees et sauvegardees.")
        for name, (psf, _, detected_center) in results.items():
            print(
                f"{name}: centre x={detected_center[0]}, y={detected_center[1]}, "
                f"taille={psf.shape[1]} x {psf.shape[0]}"
            )
        print(f"Resultats: {output_dir.resolve()}")
        return

    if args.input is None:
        raise ValueError("Fournissez --input pour estimer une PSF unique, ou utilisez --local-psf.")

    image_path = Path(args.input)
    image = load_grayscale_image(image_path)
    center = None
    if args.x is not None and args.y is not None:
        center = (args.x, args.y)
    elif args.manual_select:
        center = choose_psf_center_interactively(image)

    psf, _, detected_center = estimate_psf_from_image_file(
        image_path,
        args.kernel_size,
        center=center,
        gaussian_sigma=args.gaussian_sigma,
        downscale_factor=args.downscale_factor,
    )
    plot_psf(psf, image, detected_center, output_dir, show=args.show)

    if args.save_npy:
        np.save(output_dir / "estimated_psf.npy", psf)

    print("PSF estimee et tracee.")
    print(f"Centre utilise: x={detected_center[0]}, y={detected_center[1]}")
    print(f"Taille PSF finale: {psf.shape[1]} x {psf.shape[0]}")
    print(f"Resultats: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
