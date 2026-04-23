"""Validate YOLO dataset structure and annotation files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src import config
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.paths import list_image_files, resolve_project_path


LOGGER = get_logger(__name__)


@dataclass
class SplitReport:
    """Validation details for one dataset split."""

    split: str
    layout: str
    image_count: int = 0
    standard_label_count: int = 0
    nested_label_count: int = 0
    matched_count: int = 0
    missing_labels: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)
    duplicate_image_stems: list[str] = field(default_factory=list)
    duplicate_label_stems: list[str] = field(default_factory=list)
    class_counts: Counter[int] = field(default_factory=Counter)
    object_count: int = 0
    empty_labels: list[str] = field(default_factory=list)
    invalid_class_lines: list[str] = field(default_factory=list)
    invalid_format_lines: list[str] = field(default_factory=list)
    out_of_range_boxes: list[str] = field(default_factory=list)
    train_txt_entries: int = 0
    train_txt_existing_from_root: int = 0
    train_txt_missing_samples: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return True when the split has issues that should block training."""
        return any(
            (
                self.missing_labels,
                self.missing_images,
                self.duplicate_image_stems,
                self.duplicate_label_stems,
                self.invalid_class_lines,
                self.invalid_format_lines,
                self.out_of_range_boxes,
            )
        )


@dataclass
class DatasetReport:
    """Full dataset validation report."""

    dataset_root: Path
    layout: str
    split_reports: list[SplitReport]
    yaml_issues: list[str]
    split_overlaps: dict[str, list[str]]

    @property
    def has_errors(self) -> bool:
        """Return True when any report section contains a blocking issue."""
        return (
            any(report.has_errors for report in self.split_reports)
            or bool(self.yaml_issues)
            or bool(self.split_overlaps)
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect and validate a YOLO dataset without modifying files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=config.RAW_DATASET_DIR,
        help="Dataset root to inspect. Defaults to yolo_dataset.",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "standard", "cvat-nested"),
        default="auto",
        help="Label layout to validate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when issues are found.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum issue examples to show per category.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def _label_paths(dataset_root: Path, split: str, layout: str) -> list[Path]:
    """Return label paths for a split and layout."""
    label_root = dataset_root / "labels" / split
    if layout == "standard":
        return sorted(path for path in label_root.glob("*.txt") if path.name != "train.txt")
    if layout == "cvat-nested":
        return sorted((label_root / "labels" / "train").glob("*.txt"))
    raise ValueError(f"Unsupported layout: {layout}")


def _count_standard_labels(dataset_root: Path, split: str) -> int:
    """Count standard YOLO label files for a split."""
    label_root = dataset_root / "labels" / split
    return len([path for path in label_root.glob("*.txt") if path.name != "train.txt"])


def _count_nested_labels(dataset_root: Path, split: str) -> int:
    """Count CVAT nested label files for a split."""
    return len(list((dataset_root / "labels" / split / "labels" / "train").glob("*.txt")))


def _detect_layout(dataset_root: Path, requested_layout: str) -> str:
    """Detect the active label layout when requested."""
    if requested_layout != "auto":
        return requested_layout

    standard_count = sum(_count_standard_labels(dataset_root, split) for split in config.SPLITS)
    nested_count = sum(_count_nested_labels(dataset_root, split) for split in config.SPLITS)

    if standard_count > 0:
        return "standard"
    if nested_count > 0:
        return "cvat-nested"
    return "standard"


def _index_by_stem(paths: list[Path]) -> dict[str, list[Path]]:
    """Index files by basename without extension."""
    indexed: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        indexed[path.stem].append(path)
    return indexed


def _validate_label_file(path: Path, report: SplitReport) -> None:
    """Validate YOLO label content and update the split report."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        report.empty_labels.append(path.as_posix())
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        location = f"{path.as_posix()}:{line_number}"

        if len(parts) != 5:
            report.invalid_format_lines.append(f"{location}: {line}")
            continue

        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            report.invalid_format_lines.append(f"{location}: {line}")
            continue

        report.object_count += 1
        report.class_counts[class_id] += 1

        if class_id not in config.VALID_CLASS_IDS:
            report.invalid_class_lines.append(f"{location}: {line}")

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            report.out_of_range_boxes.append(f"{location}: {line}")


def _inspect_train_txt(dataset_root: Path, split: str, report: SplitReport) -> None:
    """Inspect CVAT train.txt references when present."""
    train_txt = dataset_root / "labels" / split / "train.txt"
    if not train_txt.exists():
        return

    for raw_line in train_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        item = raw_line.strip()
        if not item:
            continue
        report.train_txt_entries += 1
        if (dataset_root / item).exists():
            report.train_txt_existing_from_root += 1
        elif len(report.train_txt_missing_samples) < 10:
            report.train_txt_missing_samples.append(item)


def validate_split(dataset_root: Path, split: str, layout: str) -> SplitReport:
    """Validate one train, val, or test split."""
    images = list_image_files(dataset_root / "images" / split)
    labels = _label_paths(dataset_root, split, layout)
    images_by_stem = _index_by_stem(images)
    labels_by_stem = _index_by_stem(labels)

    report = SplitReport(
        split=split,
        layout=layout,
        image_count=len(images),
        standard_label_count=_count_standard_labels(dataset_root, split),
        nested_label_count=_count_nested_labels(dataset_root, split),
        matched_count=len(set(images_by_stem) & set(labels_by_stem)),
        missing_labels=sorted(set(images_by_stem) - set(labels_by_stem)),
        missing_images=sorted(set(labels_by_stem) - set(images_by_stem)),
        duplicate_image_stems=sorted(
            stem for stem, paths in images_by_stem.items() if len(paths) > 1
        ),
        duplicate_label_stems=sorted(
            stem for stem, paths in labels_by_stem.items() if len(paths) > 1
        ),
    )

    for label_path in labels:
        _validate_label_file(label_path, report)
    _inspect_train_txt(dataset_root, split, report)
    return report


def _load_yaml_file(data_yaml: Path) -> tuple[dict, str | None]:
    """Load a YAML file and return data plus an optional error message."""
    try:
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return {}, f"Could not parse {data_yaml.as_posix()}: {exc}"
    if not isinstance(data, dict):
        return {}, f"{data_yaml.as_posix()} does not contain a YAML mapping"
    return data, None


def _extract_class_names(data: dict) -> list[str]:
    """Extract class names from a YOLO data.yaml dictionary."""
    names = data.get("names")
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names or [])


def _validate_yaml_names(data_yaml: Path, issues: list[str]) -> None:
    """Validate class names in a YOLO data.yaml file."""
    data, error = _load_yaml_file(data_yaml)
    if error:
        issues.append(error)
        return

    normalized_names = _extract_class_names(data)
    if normalized_names != list(config.CLASS_NAMES):
        issues.append(
            f"{data_yaml.as_posix()} names are {normalized_names}, "
            f"expected {list(config.CLASS_NAMES)}"
        )


def validate_dataset_yaml(dataset_root: Path) -> list[str]:
    """Validate root and CVAT-exported data.yaml files when present."""
    issues: list[str] = []
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        issues.append(f"Missing root data.yaml at {data_yaml.as_posix()}")
    else:
        _validate_yaml_names(data_yaml, issues)
        data, error = _load_yaml_file(data_yaml)
        if error:
            return issues

        for key in ("train", "val"):
            if key not in data:
                issues.append(f"{data_yaml.as_posix()} is missing required key: {key}")

    for split in config.SPLITS:
        cvat_yaml = dataset_root / "labels" / split / "data.yaml"
        if cvat_yaml.exists():
            _validate_yaml_names(cvat_yaml, issues)
    return issues


def detect_split_overlaps(dataset_root: Path) -> dict[str, list[str]]:
    """Find image basename overlaps across train, val, and test splits."""
    stems_by_split = {
        split: {path.stem for path in list_image_files(dataset_root / "images" / split)}
        for split in config.SPLITS
    }
    overlaps: dict[str, list[str]] = {}
    for index, first in enumerate(config.SPLITS):
        for second in config.SPLITS[index + 1 :]:
            shared = sorted(stems_by_split[first] & stems_by_split[second])
            if shared:
                overlaps[f"{first}_vs_{second}"] = shared
    return overlaps


def validate_dataset(dataset_root: Path, requested_layout: str) -> DatasetReport:
    """Build a full validation report for a dataset root."""
    dataset_root = resolve_project_path(dataset_root)
    layout = _detect_layout(dataset_root, requested_layout)
    split_reports = [
        validate_split(dataset_root=dataset_root, split=split, layout=layout)
        for split in config.SPLITS
    ]
    return DatasetReport(
        dataset_root=dataset_root,
        layout=layout,
        split_reports=split_reports,
        yaml_issues=validate_dataset_yaml(dataset_root),
        split_overlaps=detect_split_overlaps(dataset_root),
    )


def _format_examples(values: list[str], max_examples: int) -> str:
    """Format a shortened example list."""
    if not values:
        return ""
    shown = values[:max_examples]
    suffix = "" if len(values) <= max_examples else f" ... ({len(values)} total)"
    return ", ".join(shown) + suffix


def log_report(report: DatasetReport, max_examples: int) -> None:
    """Log a human-readable validation report."""
    LOGGER.info("Dataset root: %s", report.dataset_root)
    LOGGER.info("Active label layout: %s", report.layout)

    for split_report in report.split_reports:
        LOGGER.info("[%s]", split_report.split)
        LOGGER.info("  images/%s: %d", split_report.split, split_report.image_count)
        LOGGER.info(
            "  standard labels/%s/*.txt: %d",
            split_report.split,
            split_report.standard_label_count,
        )
        LOGGER.info(
            "  nested labels/%s/labels/train/*.txt: %d",
            split_report.split,
            split_report.nested_label_count,
        )
        LOGGER.info("  matched image/label basenames: %d", split_report.matched_count)
        LOGGER.info("  objects by class id: %s", dict(sorted(split_report.class_counts.items())))
        LOGGER.info("  total labeled objects: %d", split_report.object_count)

        issue_fields = (
            ("images missing labels", split_report.missing_labels),
            ("labels missing images", split_report.missing_images),
            ("duplicate image basenames", split_report.duplicate_image_stems),
            ("duplicate label basenames", split_report.duplicate_label_stems),
            ("empty label files", split_report.empty_labels),
            ("invalid class-id lines", split_report.invalid_class_lines),
            ("invalid format lines", split_report.invalid_format_lines),
            ("out-of-range bbox lines", split_report.out_of_range_boxes),
        )
        for label, values in issue_fields:
            LOGGER.info("  %s: %d", label, len(values))
            examples = _format_examples(values, max_examples)
            if examples:
                LOGGER.info("    examples: %s", examples)

        if split_report.train_txt_entries:
            LOGGER.info("  train.txt entries: %d", split_report.train_txt_entries)
            LOGGER.info(
                "  train.txt paths existing relative to dataset root: %d",
                split_report.train_txt_existing_from_root,
            )
            examples = _format_examples(split_report.train_txt_missing_samples, max_examples)
            if examples:
                LOGGER.info("    missing train.txt examples: %s", examples)

    if report.yaml_issues:
        LOGGER.warning("data.yaml issues:")
        for issue in report.yaml_issues:
            LOGGER.warning("  %s", issue)

    if report.split_overlaps:
        LOGGER.warning("Cross-split image basename overlaps:")
        for key, values in report.split_overlaps.items():
            LOGGER.warning("  %s: %s", key, _format_examples(values, max_examples))

    if report.has_errors:
        LOGGER.warning("Dataset check finished with issues.")
    else:
        LOGGER.info("Dataset check finished without blocking issues.")


def main() -> int:
    """Run the dataset checker CLI."""
    args = parse_args()
    setup_logging(args.log_level)

    report = validate_dataset(args.dataset_root, args.layout)
    log_report(report, args.max_examples)
    return 1 if args.strict and report.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
