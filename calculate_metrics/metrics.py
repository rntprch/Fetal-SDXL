#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required. Install it first (see final dependency list)."
    ) from exc

try:
    from torchmetrics.image.kid import KernelInceptionDistance
except Exception:
    KernelInceptionDistance = None  # type: ignore[assignment]

try:
    from pytorch_msssim import ms_ssim
except Exception:
    ms_ssim = None  # type: ignore[assignment]

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # type: ignore[assignment]


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

REAL_TO_SYNTHETIC_CLASS = {
    "1_plane_trans_thalamic": "trans_thalamic",
    "2_plane_trans_cerebellum": "trans_cerebellum",
    "3_plane_trans_ventricular": "trans_ventricular",
}

if hasattr(Image, "Resampling"):
    PIL_BICUBIC = Image.Resampling.BICUBIC
else:  # pragma: no cover
    PIL_BICUBIC = Image.BICUBIC


@dataclass(frozen=True)
class ImageMeta:
    path: Path
    width: int
    height: int

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True)
class SyntheticSampleSpec:
    path: Path
    target_width: int
    target_height: int
    real_class: str
    synthetic_class: str

    @property
    def target_size(self) -> Tuple[int, int]:
        return (self.target_width, self.target_height)


@dataclass
class ClassPreparedData:
    real_class: str
    synthetic_class: str
    real_records: List[ImageMeta]
    synthetic_samples: List[SyntheticSampleSpec]


def print_info(message: str) -> None:
    print(message, flush=True)


def print_warning(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def progress_iter(
    iterable,
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    leave: bool = False,
):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)


def ensure_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def list_image_files(directory: Path) -> List[Path]:
    ensure_dir(directory, f"Directory")
    return sorted(p for p in directory.iterdir() if is_image_file(p))


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.load()
    return rgb


def resize_rgb_image(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resize target size: {(width, height)}")
    if img.size == (width, height):
        return img
    return img.resize((width, height), resample=PIL_BICUBIC)


def load_real_images_with_sizes(class_dir: Path) -> List[ImageMeta]:
    """
    Validates real images and returns metadata with original sizes.
    Broken files are skipped with warning.
    """
    ensure_dir(class_dir, f"Real class directory")
    image_paths = list_image_files(class_dir)
    records: List[ImageMeta] = []
    for path in progress_iter(
        image_paths,
        total=len(image_paths),
        desc=f"Load real: {class_dir.name}",
        leave=False,
    ):
        try:
            img = load_rgb_image(path)
            width, height = img.size
            records.append(ImageMeta(path=path, width=width, height=height))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            print_warning(f"Skipping broken real image '{path}': {exc}")
    return records


def load_synthetic_class_images(
    synthetic_root: Path,
    synthetic_class: str,
) -> List[Path]:
    class_dir = synthetic_root / synthetic_class
    ensure_dir(class_dir, f"Synthetic directory for class '{synthetic_class}'")
    return list_image_files(class_dir)


def load_and_resize_synthetic_to_real_sizes(
    synthetic_candidates: Sequence[Path],
    real_records: Sequence[ImageMeta],
    rng: random.Random,
    real_class: str,
    synthetic_class: str,
) -> List[SyntheticSampleSpec]:
    """
    Samples synthetic images and assigns each one a target real-image size.
    Resize is executed here once for validation/explicitness, then re-applied lazily later.
    Broken synthetic files are skipped with warning.
    """
    required = len(real_records)
    if required == 0:
        raise ValueError(f"Real class '{real_class}' has zero valid images after filtering.")

    if len(synthetic_candidates) < required:
        raise RuntimeError(
            "Not enough synthetic images before validation for class "
            f"'{real_class}' ({synthetic_class}): need {required}, have {len(synthetic_candidates)} "
            f"in pooled synthetic set."
        )

    shuffled = list(synthetic_candidates)
    rng.shuffle(shuffled)

    sampled: List[SyntheticSampleSpec] = []
    broken_skipped = 0
    target_index = 0

    for synthetic_path in progress_iter(
        shuffled,
        total=len(shuffled),
        desc=f"Match synth->{real_class}",
        leave=False,
    ):
        if target_index >= required:
            break

        target_size = real_records[target_index].size
        try:
            img = load_rgb_image(synthetic_path)
            _ = resize_rgb_image(img, target_size)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            broken_skipped += 1
            print_warning(f"Skipping broken synthetic image '{synthetic_path}': {exc}")
            continue

        sampled.append(
            SyntheticSampleSpec(
                path=synthetic_path,
                target_width=target_size[0],
                target_height=target_size[1],
                real_class=real_class,
                synthetic_class=synthetic_class,
            )
        )
        target_index += 1

    if len(sampled) < required:
        raise RuntimeError(
            "Not enough valid synthetic images after skipping broken files for class "
            f"'{real_class}' ({synthetic_class}): need {required}, got {len(sampled)} valid "
            f"from pooled={len(synthetic_candidates)} (broken_skipped={broken_skipped})."
        )

    return sampled


def load_resized_synthetic_image(sample: SyntheticSampleSpec) -> Image.Image:
    img = load_rgb_image(sample.path)
    return resize_rgb_image(img, sample.target_size)


def pil_to_float_tensor_rgb(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    data = data.view(h, w, 3).permute(2, 0, 1).contiguous()
    return data.float().div_(255.0)


def prepare_for_kid(
    img: Image.Image,
    kid_input_size: Tuple[int, int] = (299, 299),
) -> torch.Tensor:
    """
    Separate KID preprocessing stage.
    This may resize images for the KID backbone, after synthetic images have already
    been aligned to matched real sizes.
    """
    img = resize_rgb_image(img, kid_input_size)
    return pil_to_float_tensor_rgb(img)


def prepare_for_msssim(
    img_a: Image.Image,
    img_b: Image.Image,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Separate MS-SSIM preprocessing stage.
    Synthetic images must already be resized to matched real sizes before entering here.
    Pairwise alignment strategy: resize image B to image A size.
    """
    img_a = img_a.convert("RGB")
    img_b = img_b.convert("RGB")
    if img_b.size != img_a.size:
        img_b = resize_rgb_image(img_b, img_a.size)

    a = pil_to_float_tensor_rgb(img_a).unsqueeze(0)
    b = pil_to_float_tensor_rgb(img_b).unsqueeze(0)
    return a, b


def batched(seq: Sequence, batch_size: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def make_kid_metric(
    sample_count: int,
    device: torch.device,
    kid_subsets: int,
    kid_max_subset_size: int,
):
    if KernelInceptionDistance is None:
        raise ImportError(
            "torchmetrics is not available. Install dependencies: torchmetrics, torchvision, torch-fidelity."
        )
    if sample_count < 2:
        raise ValueError(
            f"KID requires at least 2 samples in each set, got sample_count={sample_count}."
        )
    subset_size = min(kid_max_subset_size, sample_count)
    if subset_size < 2:
        raise ValueError(
            f"KID subset_size became {subset_size}; need >= 2. sample_count={sample_count}"
        )

    try:
        metric = KernelInceptionDistance(
            feature=2048,
            subsets=kid_subsets,
            subset_size=subset_size,
            normalize=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize KernelInceptionDistance. Ensure torchmetrics + torchvision + "
            "torch-fidelity are installed and compatible."
        ) from exc

    return metric.to(device)


def kid_update_real(
    metric,
    real_records: Sequence[ImageMeta],
    batch_size: int,
    device: torch.device,
    progress_desc: Optional[str] = None,
) -> None:
    n_batches = (len(real_records) + batch_size - 1) // batch_size
    for batch in progress_iter(
        list(batched(real_records, batch_size)),
        total=n_batches,
        desc=progress_desc,
        leave=False,
    ):
        tensors: List[torch.Tensor] = []
        for record in batch:
            img = load_rgb_image(record.path)
            tensors.append(prepare_for_kid(img))
        x = torch.stack(tensors, dim=0).to(device)
        metric.update(x, real=True)


def kid_update_fake(
    metric,
    synthetic_samples: Sequence[SyntheticSampleSpec],
    batch_size: int,
    device: torch.device,
    progress_desc: Optional[str] = None,
) -> None:
    n_batches = (len(synthetic_samples) + batch_size - 1) // batch_size
    for batch in progress_iter(
        list(batched(synthetic_samples, batch_size)),
        total=n_batches,
        desc=progress_desc,
        leave=False,
    ):
        tensors: List[torch.Tensor] = []
        for sample in batch:
            img = load_resized_synthetic_image(sample)
            tensors.append(prepare_for_kid(img))
        x = torch.stack(tensors, dim=0).to(device)
        metric.update(x, real=False)


def compute_kid_from_metric(metric) -> Dict[str, float]:
    mean_t, std_t = metric.compute()
    mean_v = float(mean_t.detach().cpu().item())
    std_v = float(std_t.detach().cpu().item())
    return {"mean": mean_v, "std": std_v}


def sample_unique_index_pairs(
    n_items: int,
    requested_pairs: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    if n_items < 2 or requested_pairs <= 0:
        return []

    max_pairs = n_items * (n_items - 1) // 2
    target_pairs = min(requested_pairs, max_pairs)
    if target_pairs == 0:
        return []

    seen = set()
    pairs: List[Tuple[int, int]] = []
    while len(pairs) < target_pairs:
        i = rng.randrange(n_items)
        j = rng.randrange(n_items - 1)
        if j >= i:
            j += 1
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append((a, b))
    return pairs


def compute_msssim_for_synthetic_samples(
    samples: Sequence[SyntheticSampleSpec],
    requested_pairs: int,
    rng: random.Random,
    device: torch.device,
    progress_desc: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    if ms_ssim is None:
        raise ImportError(
            "pytorch-msssim is not available. Install dependency: pytorch-msssim"
        )

    pairs = sample_unique_index_pairs(len(samples), requested_pairs, rng)
    if not pairs:
        return {"mean": None, "std": None, "num_pairs": 0}

    scores: List[float] = []
    for i, j in progress_iter(
        pairs,
        total=len(pairs),
        desc=progress_desc,
        leave=False,
    ):
        img_a = load_resized_synthetic_image(samples[i])
        img_b = load_resized_synthetic_image(samples[j])
        a, b = prepare_for_msssim(img_a, img_b)
        a = a.to(device)
        b = b.to(device)
        with torch.no_grad():
            score = ms_ssim(a, b, data_range=1.0, size_average=True)
        scores.append(float(score.detach().cpu().item()))

    score_tensor = torch.tensor(scores, dtype=torch.float32)
    mean_v = float(score_tensor.mean().item())
    std_v = float(score_tensor.std(unbiased=False).item()) if len(scores) > 1 else 0.0
    return {"mean": mean_v, "std": std_v, "num_pairs": len(scores)}


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute KID and MS-SSIM for fetal ultrasound generation (overall + per class)."
    )
    parser.add_argument("--real_root", type=Path, required=True)
    parser.add_argument("--synthetic_root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--msssim_pairs", type=int, default=1000)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=16, help="KID loading batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for KID/MS-SSIM computation: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--kid_subsets",
        type=int,
        default=100,
        help="Number of random subsets for KID estimation.",
    )
    parser.add_argument(
        "--kid_max_subset_size",
        type=int,
        default=1000,
        help="Upper bound for KID subset_size; per run uses min(this, sample_count).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.msssim_pairs < 0:
        raise ValueError("--msssim_pairs must be >= 0")
    if args.kid_subsets <= 0:
        raise ValueError("--kid_subsets must be > 0")
    if args.kid_max_subset_size <= 1:
        raise ValueError("--kid_max_subset_size must be > 1")

    ensure_dir(args.real_root, "real_root")
    ensure_dir(args.synthetic_root, "synthetic_root")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print_info(f"[config] device={device}, seed={args.seed}")
    if tqdm is None:
        print_info("[config] tqdm is not installed; progress bars are disabled.")

    counts_real: Dict[str, int] = {}
    counts_synth_available: Dict[str, int] = {}
    counts_synth_sampled: Dict[str, int] = {}
    class_data_map: Dict[str, ClassPreparedData] = {}

    print_info("[stage] Loading real images and collecting synthetic pools...")
    for real_class, synthetic_class in REAL_TO_SYNTHETIC_CLASS.items():
        real_class_dir = args.real_root / real_class
        real_records = load_real_images_with_sizes(real_class_dir)
        if not real_records:
            raise RuntimeError(f"Real class '{real_class}' has no valid images.")
        counts_real[real_class] = len(real_records)
        print_info(f"[real] {real_class}: valid_images={len(real_records)}")

        pooled_paths = load_synthetic_class_images(args.synthetic_root, synthetic_class)
        counts_synth_available[synthetic_class] = len(pooled_paths)
        print_info(f"[synthetic] {synthetic_class}: available={len(pooled_paths)}")

        # Use a class-local RNG seeded from base seed for deterministic and stable sampling.
        class_rng = random.Random(f"{args.seed}|sample|{real_class}|{synthetic_class}")
        synthetic_samples = load_and_resize_synthetic_to_real_sizes(
            synthetic_candidates=pooled_paths,
            real_records=real_records,
            rng=class_rng,
            real_class=real_class,
            synthetic_class=synthetic_class,
        )
        counts_synth_sampled[synthetic_class] = len(synthetic_samples)
        print_info(
            f"[match] {real_class} -> {synthetic_class}: real={len(real_records)}, "
            f"sampled_synthetic={len(synthetic_samples)}"
        )

        class_data_map[real_class] = ClassPreparedData(
            real_class=real_class,
            synthetic_class=synthetic_class,
            real_records=real_records,
            synthetic_samples=synthetic_samples,
        )

    overall_real_count = sum(counts_real.values())
    overall_synth_sampled_count = sum(counts_synth_sampled.values())
    print_info(
        f"[counts] overall_real={overall_real_count}, overall_synthetic_sampled={overall_synth_sampled_count}"
    )

    if overall_real_count != overall_synth_sampled_count:
        raise RuntimeError(
            "Overall matched counts differ unexpectedly: "
            f"real={overall_real_count}, synthetic={overall_synth_sampled_count}"
        )

    print_info("[stage] Computing KID (per class + overall)...")
    overall_kid_metric = make_kid_metric(
        sample_count=overall_real_count,
        device=device,
        kid_subsets=args.kid_subsets,
        kid_max_subset_size=args.kid_max_subset_size,
    )

    kid_per_class: Dict[str, Dict[str, float]] = {}
    for real_class in REAL_TO_SYNTHETIC_CLASS:
        class_data = class_data_map[real_class]
        n = len(class_data.real_records)
        print_info(f"[kid] per-class {real_class}: n={n}")

        per_class_metric = make_kid_metric(
            sample_count=n,
            device=device,
            kid_subsets=args.kid_subsets,
            kid_max_subset_size=args.kid_max_subset_size,
        )

        with torch.no_grad():
            kid_update_real(
                per_class_metric,
                class_data.real_records,
                args.batch_size,
                device,
                progress_desc=f"KID real {real_class}",
            )
            kid_update_fake(
                per_class_metric,
                class_data.synthetic_samples,
                args.batch_size,
                device,
                progress_desc=f"KID fake {real_class}",
            )
            kid_update_real(
                overall_kid_metric,
                class_data.real_records,
                args.batch_size,
                device,
                progress_desc=f"KID overall real {real_class}",
            )
            kid_update_fake(
                overall_kid_metric,
                class_data.synthetic_samples,
                args.batch_size,
                device,
                progress_desc=f"KID overall fake {real_class}",
            )

        kid_per_class[real_class] = compute_kid_from_metric(per_class_metric)
        print_info(
            f"[kid] {real_class}: mean={kid_per_class[real_class]['mean']:.6f}, "
            f"std={kid_per_class[real_class]['std']:.6f}"
        )

    kid_overall = compute_kid_from_metric(overall_kid_metric)
    print_info(
        f"[kid] overall: mean={kid_overall['mean']:.6f}, std={kid_overall['std']:.6f}"
    )

    print_info("[stage] Computing MS-SSIM on synthetic diversity (per class + overall)...")
    msssim_per_class: Dict[str, Dict[str, Optional[float]]] = {}
    all_synthetic_samples: List[SyntheticSampleSpec] = []
    for real_class in REAL_TO_SYNTHETIC_CLASS:
        class_data = class_data_map[real_class]
        all_synthetic_samples.extend(class_data.synthetic_samples)

        pair_rng = random.Random(f"{args.seed}|msssim|{real_class}")
        print_info(
            f"[ms-ssim] per-class {real_class}: n={len(class_data.synthetic_samples)}, "
            f"requested_pairs={args.msssim_pairs}"
        )
        stats = compute_msssim_for_synthetic_samples(
            samples=class_data.synthetic_samples,
            requested_pairs=args.msssim_pairs,
            rng=pair_rng,
            device=device,
            progress_desc=f"MS-SSIM {real_class}",
        )
        msssim_per_class[real_class] = stats
        print_info(
            f"[ms-ssim] {real_class}: mean={stats['mean']}, std={stats['std']}, "
            f"num_pairs={stats['num_pairs']}"
        )

    overall_pair_rng = random.Random(f"{args.seed}|msssim|overall")
    print_info(
        f"[ms-ssim] overall: n={len(all_synthetic_samples)}, requested_pairs={args.msssim_pairs}"
    )
    msssim_overall = compute_msssim_for_synthetic_samples(
        samples=all_synthetic_samples,
        requested_pairs=args.msssim_pairs,
        rng=overall_pair_rng,
        device=device,
        progress_desc="MS-SSIM overall",
    )
    print_info(
        f"[ms-ssim] overall: mean={msssim_overall['mean']}, std={msssim_overall['std']}, "
        f"num_pairs={msssim_overall['num_pairs']}"
    )

    output = {
        "config": {
            "real_root": str(args.real_root),
            "synthetic_root": str(args.synthetic_root),
            "seed": args.seed,
            "msssim_pairs": args.msssim_pairs,
            "batch_size": args.batch_size,
            "device": str(device),
            "kid_subsets": args.kid_subsets,
            "kid_max_subset_size": args.kid_max_subset_size,
        },
        "counts": {
            "real": counts_real,
            "synthetic_available": counts_synth_available,
            "synthetic_sampled": counts_synth_sampled,
            "overall_real": overall_real_count,
            "overall_synthetic_sampled": overall_synth_sampled_count,
        },
        "kid": {
            "overall": kid_overall,
            "per_class": kid_per_class,
        },
        "ms_ssim": {
            "overall": msssim_overall,
            "per_class": msssim_per_class,
        },
    }

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print_info(f"[done] Saved JSON report to: {args.output_json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
