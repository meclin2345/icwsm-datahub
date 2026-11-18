"""Load the catalog (Parquet or CSV) with list columns restored as Python lists."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LIST_COLUMNS = (
    "authors", "author_countries", "author_regions", "platforms", "platform_families", "platform_names_verbatim",
    "collection_methods", "collection_tools", "closed_access_dependencies", "modalities", "languages_iso",
    "data_geography", "data_regions", "research_domains", "supported_tasks", "tasks_verbatim",
    "annotation_paradigms", "llms_used", "iaa_metrics", "baseline_families", "hosting_repositories",
    "dataset_urls", "data_file_formats",
)


def load_catalog(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for col in LIST_COLUMNS:
        if col not in df:
            continue
        df[col] = df[col].apply(
            lambda x: list(x) if isinstance(x, (list, np.ndarray))
            else [s for s in str(x).split("; ") if s] if isinstance(x, str) else [])
    return df


def boolcol(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].map(lambda v: str(v).lower() in ("true", "1", "1.0")).astype(int)
