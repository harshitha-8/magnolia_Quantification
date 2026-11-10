"""Compare short, matched detector runs on the existing Magnolia split."""

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import yaml
from ultralytics import YOLO


def predictions(model, root, split, device):
    files = sorted(path for path in (root / "images" / split).glob("*.jpg")
                   if not path.name.startswith("."))
    if not files:
        raise ValueError(f"No images found in {split} split")
    results = model.predict(
        source=[str(path) for path in files], conf=0.049, iou=0.45,
        imgsz=640, rect=False, device=device, verbose=False, stream=True,
    )
    for result in results:
        label = root / "labels" / split / Path(result.path).with_suffix(".txt").name
        rows = np.asarray([line.split() for line in label.read_text().splitlines() if line.strip()],
                          dtype=float).reshape(-1, 5)
        centers, half_sizes = rows[:, 1:3], rows[:, 3:5] / 2
        targets = np.column_stack((centers - half_sizes, centers + half_sizes))
        yield result.boxes.conf.cpu().numpy(), result.boxes.xyxyn.cpu().numpy(), targets


def matched_boxes(boxes, targets):
    """Match each prediction, in score order, to one unused annotation."""
    unused = list(targets)
    matched = 0
    for box in boxes:
        if not unused:
            break
        remaining = np.asarray(unused)
        overlap = np.maximum(0, np.minimum(box[2:], remaining[:, 2:]) -
                             np.maximum(box[:2], remaining[:, :2])).prod(axis=1)
        union = (box[2:] - box[:2]).prod() + (remaining[:, 2:] - remaining[:, :2]).prod(axis=1) - overlap
        ious = overlap / np.maximum(union, 1e-9)
        index = int(ious.argmax())
        if ious[index] >= 0.5:
            unused.pop(index)
            matched += 1
    return matched


def summarize(records, cutoff):
    errors = []
    detected = annotated = matched = 0
    for scores, boxes, targets in records:
        order = np.argsort(-scores)
        sorted_scores = scores[order]
        chosen = boxes[order][sorted_scores >= cutoff]
        errors.append(len(chosen) - len(targets))
        detected += len(chosen)
        annotated += len(targets)
        matched += matched_boxes(chosen, targets)
    return {
        "images": len(records),
        "confidence": cutoff,
        "mean_absolute_count_error": float(np.mean(np.abs(errors))),
        "mean_count_bias": float(np.mean(errors)),
        "detections": detected,
        "annotations": annotated,
        "matched_at_iou_0_50": matched,
        "precision_at_iou_0_50": matched / detected if detected else 0,
        "recall_at_iou_0_50": matched / annotated if annotated else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    root = Path(yaml.safe_load(args.data.read_text())["path"])
    report = {
        "protocol": "Matched pilot, COCO checkpoints, same train/validation/test split. Select each pilot cutoff by validation count MAE only, breaking ties toward higher confidence.",
        "pilot_epochs": args.epochs,
        "image_size": 640,
        "batch": args.batch,
        "device": args.device,
        "seed": 42,
        "nms_iou": 0.45,
        "pilots": {},
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    deployed = YOLO("models/magnolia_detector.pt")
    production = list(predictions(deployed, root, "test", args.device))
    report["deployed_yolo11n_30_epochs"] = [
        summarize(production, threshold) for threshold in (0.15, 0.33)
    ]
    save()

    for name in ("yolo26n", "yolo11n", "yolo11m"):
        started = monotonic()
        model = YOLO(str(args.weights_dir / f"{name}.pt"))
        model.train(
            data=str(args.data), epochs=args.epochs, imgsz=640, batch=args.batch,
            device=args.device, workers=0, project=str(args.work_dir), name=name,
            seed=42, deterministic=True, patience=0, cache=False, amp=False,
            mosaic=0.5, close_mosaic=2, fliplr=0.5, flipud=0.5, degrees=15,
            translate=0.1, scale=0.3, plots=False, exist_ok=False,
        )
        trained = YOLO(str(model.trainer.best))
        validation = list(predictions(trained, root, "val", args.device))
        candidates = [summarize(validation, cutoff / 100) for cutoff in range(5, 61, 5)]
        selected = min(candidates, key=lambda row: (row["mean_absolute_count_error"], -row["confidence"]))
        test = list(predictions(trained, root, "test", args.device))
        report["pilots"][name] = {
            "seconds": round(monotonic() - started, 1),
            "validation": selected,
            "test": summarize(test, selected["confidence"]),
        }
        save()
        print(f"Comparison saved: {name}: {report['pilots'][name]['test']}", flush=True)


if __name__ == "__main__":
    main()
