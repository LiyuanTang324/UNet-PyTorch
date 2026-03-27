import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from unet import UNet


ALL_DATASETS = [
    "BUS-BRA",
    "BUSI",
    "BUSIS",
    "CAMUS",
    "DDTI",
    "Fetal_HC",
    "KidneyUS",
]


def parse_dataset_config(config_path: Path) -> Dict[int, int]:
    value_to_class: Dict[int, int] = {}
    lines = config_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        # expected format: class_idx:class_name:pixel_value
        parts = line.split(":")
        if len(parts) != 3:
            raise ValueError(f"Unexpected config format in {config_path}: {line}")
        class_idx = int(parts[0].strip())
        pixel_value = int(parts[2].strip())
        value_to_class[pixel_value] = class_idx
    return value_to_class


def center_crop_target(target: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    _, h, w = target.shape
    dh = (h - out_h) // 2
    dw = (w - out_w) // 2
    return target[:, dh : dh + out_h, dw : dw + out_w]


def to_class_mask(mask: np.ndarray, value_to_class: Dict[int, int]) -> np.ndarray:
    class_mask = np.zeros_like(mask, dtype=np.int64)
    for pixel_value, class_idx in value_to_class.items():
        class_mask[mask == pixel_value] = class_idx
    return class_mask


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("L").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def mask_to_tensor(mask: Image.Image, image_size: int, value_to_class: Dict[int, int]) -> torch.Tensor:
    mask = mask.convert("L").resize((image_size, image_size), Image.NEAREST)
    mask_np = np.asarray(mask, dtype=np.uint8)
    class_mask = to_class_mask(mask_np, value_to_class)
    return torch.from_numpy(class_mask).long()


def parse_split_line(line: str) -> Tuple[str, Optional[str]]:
    # Supports:
    # 1) "image_name.png"
    # 2) "image_name.png mask_name.png"
    # 3) "image_name.png,mask_name.png"
    line = line.strip()
    if not line:
        raise ValueError("Empty split line.")
    if "," in line:
        items = [x.strip() for x in line.split(",") if x.strip()]
    else:
        items = [x.strip() for x in line.split() if x.strip()]
    if len(items) == 1:
        return items[0], None
    if len(items) >= 2:
        return items[0], items[1]
    raise ValueError(f"Cannot parse split line: {line}")


def auto_detect_subdir(root: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = root / name
        if p.exists() and p.is_dir():
            return p
    return None


class SegmentationDataset(Dataset):
    def __init__(
        self,
        split_file: Path,
        image_dir: Path,
        mask_dir: Path,
        image_size: int,
        value_to_class: Dict[int, int],
        mask_suffix: str = "",
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.value_to_class = value_to_class
        self.mask_suffix = mask_suffix
        self.records: List[Tuple[str, Optional[str]]] = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.records.append(parse_split_line(line))
        if len(self.records) == 0:
            raise ValueError(f"No samples found in split file: {split_file}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        image_name, mask_name = self.records[idx]
        image_path = self.image_dir / image_name
        if mask_name is None:
            stem, ext = Path(image_name).stem, Path(image_name).suffix
            mask_name = f"{stem}{self.mask_suffix}{ext}"
        mask_path = self.mask_dir / mask_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        image_t = image_to_tensor(image, self.image_size)
        mask_t = mask_to_tensor(mask, self.image_size, self.value_to_class)
        return image_t, mask_t, image_name


def compute_binary_metrics_per_class(
    pred_np: np.ndarray,
    gt_np: np.ndarray,
    cls_idx: int,
    eps: float = 1e-7,
) -> Tuple[float, float, float]:
    pred_c = pred_np == cls_idx
    gt_c = gt_np == cls_idx
    intersection = float(np.logical_and(pred_c, gt_c).sum())
    pred_sum = float(pred_c.sum())
    gt_sum = float(gt_c.sum())
    union = float(np.logical_or(pred_c, gt_c).sum())

    if pred_sum == 0.0 and gt_sum == 0.0:
        dice = 1.0
        iou = 1.0
        hd95 = 0.0
        return dice, iou, hd95

    dice = (2.0 * intersection + eps) / (pred_sum + gt_sum + eps)
    iou = (intersection + eps) / (union + eps)

    if pred_sum > 0.0 and gt_sum > 0.0:
        hd95 = float(metric.binary.hd95(pred_c, gt_c))
    else:
        hd95 = float("inf")
    return dice, iou, hd95


@dataclass
class EvalOutput:
    loss: float
    dsc: float
    iou: float
    hd95: float
    fps: Optional[float] = None
    flops_g: Optional[float] = None
    vram_peak_gb: Optional[float] = None


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    device: torch.device,
    measure_perf: bool = False,
) -> EvalOutput:
    model.eval()
    total_loss = 0.0
    dsc_list: List[float] = []
    iou_list: List[float] = []
    hd95_list: List[float] = []

    infer_time = 0.0
    infer_count = 0

    if measure_perf and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if measure_perf and device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            logits = model(images)
            if measure_perf and device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            infer_time += t1 - t0
            infer_count += images.size(0)

            targets_cropped = center_crop_target(targets, logits.shape[-2], logits.shape[-1])
            loss = criterion(logits, targets_cropped)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            preds_np = preds.detach().cpu().numpy()
            targets_np = targets_cropped.detach().cpu().numpy()
            for b in range(preds_np.shape[0]):
                sample_dice = []
                sample_iou = []
                sample_hd95 = []
                for cls_idx in range(1, num_classes):
                    dsc, iou, hd95 = compute_binary_metrics_per_class(
                        preds_np[b], targets_np[b], cls_idx
                    )
                    sample_dice.append(dsc)
                    sample_iou.append(iou)
                    if np.isfinite(hd95):
                        sample_hd95.append(hd95)
                dsc_list.append(float(np.mean(sample_dice)) if sample_dice else 1.0)
                iou_list.append(float(np.mean(sample_iou)) if sample_iou else 1.0)
                hd95_list.append(float(np.mean(sample_hd95)) if sample_hd95 else 0.0)

    size = len(loader.dataset)
    avg_loss = total_loss / max(size, 1)
    output = EvalOutput(
        loss=avg_loss,
        dsc=float(np.mean(dsc_list)) if dsc_list else 0.0,
        iou=float(np.mean(iou_list)) if iou_list else 0.0,
        hd95=float(np.mean(hd95_list)) if hd95_list else 0.0,
    )

    if measure_perf and infer_time > 0:
        output.fps = infer_count / infer_time
        if device.type == "cuda":
            output.vram_peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    return output


def estimate_flops_g(model: nn.Module, image_size: int, device: torch.device) -> Optional[float]:
    try:
        from thop import profile
    except Exception:
        return None

    model.eval()
    dummy = torch.randn(1, 1, image_size, image_size, device=device)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    flops = 2.0 * float(macs)
    return flops / 1e9


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        targets_cropped = center_crop_target(targets, logits.shape[-2], logits.shape[-1])
        loss = criterion(logits, targets_cropped)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / max(len(loader.dataset), 1)


def get_dataset_dirs(dataset_root: Path, image_dir_name: Optional[str], mask_dir_name: Optional[str]) -> Tuple[Path, Path]:
    if image_dir_name is not None:
        image_dir = dataset_root / image_dir_name
    else:
        image_dir = auto_detect_subdir(dataset_root, ["images", "image", "imgs", "img"])
    if mask_dir_name is not None:
        mask_dir = dataset_root / mask_dir_name
    else:
        mask_dir = auto_detect_subdir(dataset_root, ["masks", "mask", "labels", "label", "annotations", "gt"])

    if image_dir is None or not image_dir.exists():
        raise FileNotFoundError(
            f"Cannot resolve image dir in {dataset_root}. "
            "Set --image-dir-name explicitly."
        )
    if mask_dir is None or not mask_dir.exists():
        raise FileNotFoundError(
            f"Cannot resolve mask dir in {dataset_root}. "
            "Set --mask-dir-name explicitly."
        )
    return image_dir, mask_dir


def run_single_dataset(args, dataset_name: str, device: torch.device) -> None:
    dataset_root = Path(args.data_root) / dataset_name
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    config_path = dataset_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    value_to_class = parse_dataset_config(config_path)
    num_classes = max(value_to_class.values()) + 1
    image_dir, mask_dir = get_dataset_dirs(dataset_root, args.image_dir_name, args.mask_dir_name)

    train_ds = SegmentationDataset(
        split_file=dataset_root / "train.txt",
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=args.image_size,
        value_to_class=value_to_class,
        mask_suffix=args.mask_suffix,
    )
    val_ds = SegmentationDataset(
        split_file=dataset_root / "val.txt",
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=args.image_size,
        value_to_class=value_to_class,
        mask_suffix=args.mask_suffix,
    )
    test_ds = SegmentationDataset(
        split_file=dataset_root / "test.txt",
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=args.image_size,
        value_to_class=value_to_class,
        mask_suffix=args.mask_suffix,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = UNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.output_root) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best_model.pth"

    history = []
    best_val_dsc = -1.0
    best_epoch = -1

    epoch_iter = tqdm(range(1, args.epochs + 1), desc=f"{dataset_name} training")
    for epoch in epoch_iter:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_output = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            num_classes=num_classes,
            device=device,
            measure_perf=False,
        )

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_output.loss,
            "val_dsc": val_output.dsc,
            "val_iou": val_output.iou,
            "val_hd95": val_output.hd95,
        }
        history.append(epoch_log)
        epoch_iter.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_dsc=f"{val_output.dsc:.4f}",
            val_iou=f"{val_output.iou:.4f}",
        )

        # Select best model by validation DSC.
        if val_output.dsc > best_val_dsc:
            best_val_dsc = val_output.dsc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "dataset": dataset_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dsc": val_output.dsc,
                    "num_classes": num_classes,
                    "image_size": args.image_size,
                },
                best_ckpt,
            )

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_output = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        num_classes=num_classes,
        device=device,
        measure_perf=True,
    )
    test_output.flops_g = estimate_flops_g(model, args.image_size, device)

    result = {
        "dataset": dataset_name,
        "best_epoch_by_val_dsc": best_epoch,
        "best_val_dsc": best_val_dsc,
        "test_metrics": {
            "dsc": test_output.dsc,
            "iou": test_output.iou,
            "hd95": test_output.hd95,
            "fps": test_output.fps,
            "flops_g": test_output.flops_g,
            "vram_peak_gb": test_output.vram_peak_gb,
        },
        "history": history,
    }

    with (save_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[{dataset_name}] best_epoch={best_epoch}, val_dsc={best_val_dsc:.4f}")
    print(f"[{dataset_name}] test={json.dumps(result['test_metrics'], ensure_ascii=False)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="U-Net training/testing on 7 segmentation datasets with DSC/IoU/HD95 and FPS/FLOPs/VRAM evaluation."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "segmentation"),
        help="Root directory containing 7 segmentation dataset folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        help="Datasets to run. Default is all 7 segmentation datasets.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--image-size", type=int, default=572, help="Input size for U-Net.")
    parser.add_argument("--output-root", type=str, default="./runs_unet")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--image-dir-name",
        type=str,
        default=None,
        help="Image subdir name under each dataset folder; auto-detected if omitted.",
    )
    parser.add_argument(
        "--mask-dir-name",
        type=str,
        default=None,
        help="Mask subdir name under each dataset folder; auto-detected if omitted.",
    )
    parser.add_argument(
        "--mask-suffix",
        type=str,
        default="",
        help="Mask suffix when split txt only contains image names, e.g. '_mask'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    unknown = [d for d in args.datasets if d not in ALL_DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Allowed: {ALL_DATASETS}")

    for dataset_name in args.datasets:
        run_single_dataset(args, dataset_name, device)


if __name__ == "__main__":
    main()
