"""Create a clean YOLO dataset copy from the current CVAT-style export."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from src import config
from src.check_dataset import detect_split_overlaps
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.paths import as_posix_path, ensure_directory, list_image_files, resolve_project_path


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class CopyPlanItem:
    """One image/label pair to copy into the prepared dataset."""

    split: str
    image_source: Path
    image_target: Path
    label_source: Path | None
    label_target: Path
    create_empty_label: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare a standard Ultralytics YOLO dataset without deleting source data."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=config.RAW_DATASET_DIR,
        help="Raw dataset root. Defaults to yolo_dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=config.PREPARED_DATASET_DIR,
        help="Prepared dataset output root. Defaults to yolo_dataset/prepared.",
    )
    parser.add_argument(
        "--source-layout",
        choices=("standard", "cvat-nested"),
        default="cvat-nested",
        help="Where to read labels from.",
    )
    parser.add_argument(
        "--include-unlabeled",
        choices=("exclude", "empty"),
        default="exclude",
        help=(
            "How to handle images with no label file. Use 'empty' only for intentional "
            "background-only images."
        ),
    )
    parser.add_argument(
        "--overlap-policy",
        choices=("keep", "drop-from-train"),
        default="keep",
        help=(
            "How to handle cross-split basename overlaps in the prepared copy. "
            "The default keeps them and only warns."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually copy files. Without this flag the command is a dry run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared files when copying.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def _source_label_dir(source_root: Path, split: str, source_layout: str) -> Path:
    """Return the label directory for the requested source layout."""
    if source_layout == "standard":
        return source_root / "labels" / split
    if source_layout == "cvat-nested":
        return source_root / "labels" / split / "labels" / "train"
    raise ValueError(f"Unsupported source layout: {source_layout}")


def _label_index(source_root: Path, split: str, source_layout: str) -> dict[str, Path]:
    """Build a basename-to-label index for a split."""
    label_dir = _source_label_dir(source_root, split, source_layout)
    labels = sorted(path for path in label_dir.glob("*.txt") if path.name != "train.txt")
    return {path.stem: path for path in labels}


def build_copy_plan(
    source_root: Path,
    output_root: Path,
    source_layout: str,
    include_unlabeled: str,
    excluded_stems_by_split: dict[str, set[str]] | None = None,
) -> list[CopyPlanItem]:
    """Build the non-destructive copy plan for all dataset splits."""
    plan: list[CopyPlanItem] = []
    excluded_stems_by_split = excluded_stems_by_split or {}

    for split in config.SPLITS:
        images = list_image_files(source_root / "images" / split)
        labels_by_stem = _label_index(source_root, split, source_layout)
        excluded_stems = excluded_stems_by_split.get(split, set())

        for image_path in images:
            if image_path.stem in excluded_stems:
                continue

            label_path = labels_by_stem.get(image_path.stem)
            if label_path is None and include_unlabeled == "exclude":
                continue

            plan.append(
                CopyPlanItem(
                    split=split,
                    image_source=image_path,
                    image_target=output_root / "images" / split / image_path.name,
                    label_source=label_path,
                    label_target=output_root / "labels" / split / f"{image_path.stem}.txt",
                    create_empty_label=label_path is None,
                )
            )

    return plan


def build_overlap_exclusions(
    source_root: Path,
    overlap_policy: str,
) -> dict[str, set[str]]:
    """Build split-specific stem exclusions for the prepared copy."""
    if overlap_policy == "keep":
        return {}
    if overlap_policy != "drop-from-train":
        raise ValueError(f"Unsupported overlap policy: {overlap_policy}")

    overlaps = detect_split_overlaps(source_root)
    train_exclusions: set[str] = set()
    for key, stems in overlaps.items():
        if key.startswith("train_vs_"):
            train_exclusions.update(stems)
    return {"train": train_exclusions} if train_exclusions else {}


def _copy_file(source: Path, target: Path, overwrite: bool) -> None:
    """Copy a file while respecting overwrite safety."""
    if target.exists() and not overwrite:
        LOGGER.debug("Skipping existing file: %s", target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def apply_copy_plan(plan: list[CopyPlanItem], overwrite: bool) -> None:
    """Copy images and labels according to the prepared copy plan."""
    for item in plan:
        _copy_file(item.image_source, item.image_target, overwrite=overwrite)
        item.label_target.parent.mkdir(parents=True, exist_ok=True)
        if item.create_empty_label:
            if item.label_target.exists() and not overwrite:
                LOGGER.debug("Skipping existing empty label: %s", item.label_target)
                continue
            item.label_target.write_text("", encoding="utf-8")
        elif item.label_source is not None:
            _copy_file(item.label_source, item.label_target, overwrite=overwrite)


def write_dataset_yaml(output_root: Path) -> Path:
    """Write an Ultralytics-compatible data.yaml for the prepared dataset."""
    data = {
        "path": as_posix_path(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(config.CLASS_NAMES)},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml


def log_plan_summary(
    plan: list[CopyPlanItem],
    include_unlabeled: str,
    overlap_policy: str,
    exclusions: dict[str, set[str]],
    apply: bool,
) -> None:
    """Log a concise copy-plan summary."""
    LOGGER.info("Mode: %s", "apply" if apply else "dry-run")
    LOGGER.info("Unlabeled image handling: %s", include_unlabeled)
    LOGGER.info("Overlap policy: %s", overlap_policy)
    for split, stems in sorted(exclusions.items()):
        LOGGER.info("%s: %d overlapping basenames excluded", split, len(stems))

    for split in config.SPLITS:
        split_items = [item for item in plan if item.split == split]
        empty_labels = sum(1 for item in split_items if item.create_empty_label)
        LOGGER.info(
            "%s: %d images planned, %d empty labels planned",
            split,
            len(split_items),
            empty_labels,
        )


def main() -> int:
    """Run the dataset preparation CLI."""
    args = parse_args()
    setup_logging(args.log_level)

    source_root = resolve_project_path(args.source_root)
    output_root = resolve_project_path(args.output_root)

    LOGGER.info("Source root: %s", source_root)
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Source label layout: %s", args.source_layout)

    overlaps = detect_split_overlaps(source_root)
    if overlaps:
        LOGGER.warning("Detected cross-split basename overlaps in the source dataset.")
        for key, values in overlaps.items():
            shown = ", ".join(values[:10])
            suffix = "" if len(values) <= 10 else f" ... ({len(values)} total)"
            LOGGER.warning("  %s: %s%s", key, shown, suffix)

    exclusions = build_overlap_exclusions(source_root, args.overlap_policy)
    plan = build_copy_plan(
        source_root=source_root,
        output_root=output_root,
        source_layout=args.source_layout,
        include_unlabeled=args.include_unlabeled,
        excluded_stems_by_split=exclusions,
    )
    log_plan_summary(
        plan,
        include_unlabeled=args.include_unlabeled,
        overlap_policy=args.overlap_policy,
        exclusions=exclusions,
        apply=args.apply,
    )

    if not args.apply:
        LOGGER.info("Dry run only. Re-run with --apply to create the prepared dataset.")
        return 0

    ensure_directory(output_root)
    apply_copy_plan(plan, overwrite=args.overwrite)
    data_yaml = write_dataset_yaml(output_root)
    LOGGER.info("Prepared dataset written to: %s", output_root)
    LOGGER.info("Dataset YAML written to: %s", data_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
