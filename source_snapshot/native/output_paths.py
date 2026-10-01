from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

OUTPUT_SUBDIRS = {
    "train": "train",
    "logs": "logs",
    "evaluation": "evaluation",
    "summaries": "summaries",
}


def output_path(kind: str, *parts) -> Path:
    """Return a path under outputs/ for generated experiment artifacts."""
    if kind not in OUTPUT_SUBDIRS:
        raise ValueError(f"Unknown output kind: {kind}")
    return OUTPUT_ROOT.joinpath(OUTPUT_SUBDIRS[kind], *map(str, parts))


def normalize_output_path(path, kind: str) -> Path:
    """Prefix a relative generated-output path with outputs/<kind>/."""
    path = Path(path)
    if path.is_absolute() or path.parts[:1] == ("outputs",):
        return path
    return output_path(kind, path)


def resolve_train_path(path) -> Path:
    """Find a training artifact path before or after consolidation."""
    path = Path(path)
    if path.exists():
        return path
    candidate = output_path("train", path)
    if candidate.exists():
        return candidate
    return path
