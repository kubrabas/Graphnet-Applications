#!/usr/bin/env python3
"""Build the deterministic 340StringMC master train/validation/test manifest."""

# Slurm usage:
# sbatch \
#   --account=def-nahee \
#   --job-name=MasterSplit \
#   --time=00:30:00 \
#   --mem=32G \
#   --cpus-per-task=4 \
#   --output=/project/def-nahee/kbas/Graphnet-Applications/Metadata/DatasetSplits/340StringMC/master_split_slurm_%j.log \
#   --wrap="/project/def-nahee/kbas/.venv_try/bin/python /project/def-nahee/kbas/Graphnet-Applications/DataPreperation/Parquet/build_master_split.py"

import argparse
import hashlib
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


DEFAULT_TRUTH_DIR = Path(
    "/project/def-nahee/kbas/Graphnet-Applications/Metadata/"
    "DatasetStatistics/TruthLevelStatistics"
)
DEFAULT_OUTPUT_DIR = Path(
    "/project/def-nahee/kbas/Graphnet-Applications/Metadata/"
    "DatasetSplits/340StringMC"
)
OUTPUT_BASENAME = "master_split_emax1e6"
FLAVORS = ("Muon", "Electron", "Tau", "NC")
EVENT_COLUMNS = ["RunID", "SubrunID", "EventID", "SubEventID"]
EVENT_KEY_COLUMNS = ["flavor", *EVENT_COLUMNS]
TRIGGER_COLUMNS = [
    "triggered_nonoise_102_string",
    "triggered_nonoise_160_string",
    "triggered_nonoise_340_string",
]
TRIGGER_RENAME = {
    "triggered_nonoise_102_string": "trigger_102",
    "triggered_nonoise_160_string": "trigger_160",
    "triggered_nonoise_340_string": "trigger_340",
}
STRATIFY_COLUMNS = [
    "flavor",
    "trigger_tier",
    "energy_bin",
    "cos_zenith_bin",
]
SPLIT_FRACTIONS = {"train": 0.80, "validation": 0.10, "test": 0.10}
SPLIT_ORDER = ["train", "validation", "test"]
MANIFEST_COLUMNS = [
    *EVENT_KEY_COLUMNS,
    "split",
    "stratum_id",
    "trigger_102",
    "trigger_160",
    "trigger_340",
]
DEFAULT_HASH_KEY = "pone_split_00042"
DEFAULT_EXPECTED_STRATA = 96
DEFAULT_EXPECTED_ENERGY_EDGES = [2.0, 3.0, 4.0, 5.0, 6.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic 340StringMC master split manifest after "
            "validating all candidate strata."
        )
    )
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=DEFAULT_TRUTH_DIR,
        help="Directory containing Muon/Electron/Tau/NC truth CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for timestamped parquet and log outputs.",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=1e6,
        help="Keep events with totalEnergy strictly below this value in GeV.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows per pandas CSV chunk.",
    )
    parser.add_argument(
        "--minimum-stratum-size",
        type=int,
        default=10,
        help="Fail when any candidate stratum contains fewer events.",
    )
    parser.add_argument(
        "--expected-strata",
        type=int,
        default=DEFAULT_EXPECTED_STRATA,
        help="Expected number of candidate strata.",
    )
    parser.add_argument(
        "--expected-energy-edges",
        type=float,
        nargs="+",
        default=DEFAULT_EXPECTED_ENERGY_EDGES,
        help="Expected log10(GeV) energy-bin edges.",
    )
    parser.add_argument(
        "--hash-key",
        default=DEFAULT_HASH_KEY,
        help="Exactly 16-character pandas hash key for deterministic assignment.",
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_master_split")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.FileHandler(log_path, mode="x"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_table(logger: logging.Logger, title: str, table: pd.DataFrame) -> None:
    logger.info("%s\n%s", title, table.to_string())


def load_selected_truth(
    truth_dir: Path,
    energy_max: float,
    chunksize: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    use_columns = EVENT_COLUMNS + ["totalEnergy", "zenith"] + TRIGGER_COLUMNS
    flavor_frames = []

    for flavor in FLAVORS:
        csv_path = truth_dir / f"{flavor}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing truth CSV: {csv_path}")

        selected_chunks = []
        rows_read = 0
        for chunk in pd.read_csv(
            csv_path,
            usecols=use_columns,
            chunksize=chunksize,
        ):
            rows_read += len(chunk)
            trigger_mask = chunk[TRIGGER_COLUMNS].eq(1).any(axis=1)
            selection_mask = trigger_mask & chunk["totalEnergy"].lt(energy_max)
            selected = chunk.loc[selection_mask].copy()
            if not selected.empty:
                selected_chunks.append(selected)

        if not selected_chunks:
            raise ValueError(
                f"{flavor} has no events below {energy_max:g} GeV that trigger "
                "in at least one geometry."
            )

        flavor_df = pd.concat(selected_chunks, ignore_index=True)
        flavor_df.insert(0, "flavor", flavor)
        flavor_frames.append(flavor_df)
        logger.info(
            "%-8s: read=%s selected=%s (totalEnergy < %g and triggered)",
            flavor,
            f"{rows_read:,}",
            f"{len(flavor_df):,}",
            energy_max,
        )

    master_df = pd.concat(flavor_frames, ignore_index=True).rename(
        columns=TRIGGER_RENAME
    )
    for column in TRIGGER_RENAME.values():
        master_df[column] = master_df[column].eq(1)

    master_df = master_df[
        [
            "flavor",
            *EVENT_COLUMNS,
            "totalEnergy",
            "zenith",
            "trigger_102",
            "trigger_160",
            "trigger_340",
        ]
    ]
    logger.info(
        "Master dataframe: %s events x %s columns",
        f"{len(master_df):,}",
        master_df.shape[1],
    )
    return master_df


def build_candidate_strata(
    master_df: pd.DataFrame,
    minimum_size: int,
    expected_strata: int,
    expected_energy_edges: list[float],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    required_columns = {
        "flavor",
        "totalEnergy",
        "zenith",
        "trigger_102",
        "trigger_160",
        "trigger_340",
    }
    missing_columns = required_columns.difference(master_df.columns)
    if missing_columns:
        raise KeyError(
            f"master_df is missing required columns: {sorted(missing_columns)}"
        )
    if master_df[list(required_columns)].isna().any().any():
        raise ValueError("Required stratification columns contain missing values.")
    if (master_df["totalEnergy"] <= 0).any():
        raise ValueError("totalEnergy must be positive before taking log10.")

    non_nested = (
        master_df["trigger_102"] & ~master_df["trigger_160"]
    ) | (master_df["trigger_160"] & ~master_df["trigger_340"])
    if non_nested.any():
        raise ValueError(
            f"Found {int(non_nested.sum()):,} events that violate the nested "
            "trigger structure."
        )
    logger.info("Nested trigger structure: valid")

    trigger_tier_order = ["102", "160_not_102", "340_not_160"]
    master_df["trigger_tier"] = pd.Categorical(
        np.select(
            [master_df["trigger_102"], master_df["trigger_160"]],
            ["102", "160_not_102"],
            default="340_not_160",
        ),
        categories=trigger_tier_order,
        ordered=True,
    )

    master_df["log10_energy"] = np.log10(master_df["totalEnergy"])
    observed_loge_min = master_df["log10_energy"].min()
    observed_loge_max = master_df["log10_energy"].max()
    energy_min = np.floor(observed_loge_min + 1e-10)
    energy_max = np.ceil(observed_loge_max - 1e-10)
    energy_edges = np.arange(energy_min, energy_max + 0.5, 1.0)
    energy_edges[0] = min(
        energy_edges[0],
        np.nextafter(observed_loge_min, -np.inf),
    )
    energy_edges[-1] = max(
        energy_edges[-1],
        np.nextafter(observed_loge_max, np.inf),
    )
    master_df["energy_bin"] = pd.cut(
        master_df["log10_energy"],
        bins=energy_edges,
        include_lowest=True,
    )
    if master_df["energy_bin"].isna().any():
        raise ValueError("Some events were not assigned to an energy bin.")

    master_df["cos_zenith"] = np.cos(master_df["zenith"]).clip(-1.0, 1.0)
    master_df["cos_zenith_bin"] = pd.cut(
        master_df["cos_zenith"],
        bins=[-1.0, 0.0, 1.0],
        include_lowest=True,
    )
    if master_df["cos_zenith_bin"].isna().any():
        raise ValueError("Some events were not assigned to a cos(zenith) bin.")

    stratum_groupby = master_df.groupby(
        STRATIFY_COLUMNS,
        observed=True,
        sort=True,
    )
    master_df["stratum_id"] = stratum_groupby.ngroup()
    stratum_sizes = (
        master_df.groupby(
            ["stratum_id", *STRATIFY_COLUMNS],
            observed=True,
            sort=True,
        )
        .size()
        .rename("n_events")
        .reset_index()
        .sort_values("n_events")
    )
    small_strata = stratum_sizes[
        stratum_sizes["n_events"] < minimum_size
    ]
    rounded_edges = np.round(energy_edges, 3).tolist()

    logger.info(
        "Observed candidate strata: %s",
        f"{len(stratum_sizes):,}",
    )
    logger.info(
        "Observed smallest stratum: %s events",
        f"{int(stratum_sizes['n_events'].min()):,}",
    )
    logger.info("Required minimum stratum size: %s events", minimum_size)
    logger.info(
        "Strata with fewer than %s events: %s",
        minimum_size,
        f"{len(small_strata):,}",
    )
    logger.info("Energy-bin edges in log10(GeV): %s", rounded_edges)
    log_table(
        logger,
        "Trigger tier x flavor:",
        pd.crosstab(
            master_df["trigger_tier"],
            master_df["flavor"],
            margins=True,
        ),
    )
    log_table(
        logger,
        "Stratum size distribution:",
        stratum_sizes["n_events"]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_frame(),
    )

    failures = []
    if len(stratum_sizes) != expected_strata:
        failures.append(
            f"expected {expected_strata} candidate strata, found "
            f"{len(stratum_sizes)}"
        )
    expected_edges = np.asarray(expected_energy_edges, dtype=float)
    edges_match = energy_edges.shape == expected_edges.shape and np.allclose(
        energy_edges,
        expected_edges,
        rtol=0.0,
        atol=1e-9,
    )
    if not edges_match:
        failures.append(
            f"expected energy edges {expected_energy_edges}, found "
            f"{rounded_edges}"
        )
    if not small_strata.empty:
        failures.append(
            f"{len(small_strata)} strata contain fewer than "
            f"{minimum_size} events"
        )
        log_table(
            logger,
            "Smallest candidate strata:",
            small_strata.head(30),
        )

    sizes = stratum_sizes["n_events"].to_numpy(dtype=np.int64)
    split_allocations = np.column_stack(
        [
            np.floor(SPLIT_FRACTIONS["train"] * sizes).astype(np.int64),
            np.floor(SPLIT_FRACTIONS["validation"] * sizes).astype(np.int64),
        ]
    )
    test_sizes = sizes - split_allocations.sum(axis=1)
    if (
        (split_allocations[:, 0] <= 0).any()
        or (split_allocations[:, 1] <= 0).any()
        or (test_sizes <= 0).any()
    ):
        failures.append(
            "at least one candidate stratum cannot populate train, "
            "validation, and test"
        )

    if failures:
        raise RuntimeError(
            "Candidate-strata preflight failed: " + "; ".join(failures)
        )

    logger.info(
        "All candidate strata are large enough for an 80/10/10 split."
    )
    logger.info("Candidate-strata preflight: PASSED")
    return master_df, stratum_sizes, energy_edges


def assign_splits(
    master_df: pd.DataFrame,
    hash_key: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    if len(hash_key) != 16:
        raise ValueError("--hash-key must contain exactly 16 characters.")
    if not master_df.index.is_unique:
        raise ValueError("master_df index must be unique before assigning splits.")

    duplicate_event_keys = master_df.duplicated(
        EVENT_KEY_COLUMNS,
        keep=False,
    )
    if duplicate_event_keys.any():
        duplicate_examples = master_df.loc[
            duplicate_event_keys,
            EVENT_KEY_COLUMNS,
        ].head(20)
        log_table(logger, "Duplicate event-key examples:", duplicate_examples)
        raise ValueError(
            f"Found {int(duplicate_event_keys.sum()):,} rows with duplicated "
            "physical event keys."
        )

    master_df["split_hash"] = pd.util.hash_pandas_object(
        master_df[EVENT_KEY_COLUMNS],
        index=False,
        hash_key=hash_key,
    ).astype("uint64")
    ranked = master_df.sort_values(
        ["stratum_id", "split_hash", *EVENT_KEY_COLUMNS],
        kind="mergesort",
    )
    rank_within_stratum = (
        ranked.groupby("stratum_id", sort=False).cumcount().to_numpy()
    )
    size_of_stratum = (
        ranked.groupby("stratum_id", sort=False)["stratum_id"]
        .transform("size")
        .to_numpy()
    )
    n_train = np.floor(
        SPLIT_FRACTIONS["train"] * size_of_stratum
    ).astype("int64")
    n_validation = np.floor(
        SPLIT_FRACTIONS["validation"] * size_of_stratum
    ).astype("int64")
    split_values = np.where(
        rank_within_stratum < n_train,
        "train",
        np.where(
            rank_within_stratum < n_train + n_validation,
            "validation",
            "test",
        ),
    )
    split_by_index = pd.Series(split_values, index=ranked.index)
    master_df["split"] = pd.Categorical(
        split_by_index.reindex(master_df.index),
        categories=SPLIT_ORDER,
        ordered=True,
    )

    if master_df["split"].isna().any():
        raise AssertionError("Some events did not receive a split assignment.")
    stratum_split_counts = pd.crosstab(
        master_df["stratum_id"],
        master_df["split"],
    ).reindex(columns=SPLIT_ORDER, fill_value=0)
    if (stratum_split_counts == 0).any().any():
        raise AssertionError(
            "At least one stratum is missing from train, validation, or test."
        )

    split_summary = (
        master_df.groupby("split", observed=False)
        .size()
        .rename("n_events")
        .to_frame()
    )
    split_summary["fraction"] = split_summary["n_events"] / len(master_df)
    flavor_split_counts = pd.crosstab(
        master_df["split"],
        master_df["flavor"],
        margins=True,
    )
    geometry_split_counts = pd.DataFrame(
        {
            "102 strings": master_df.loc[
                master_df["trigger_102"]
            ].groupby("split", observed=False).size(),
            "160 strings": master_df.loc[
                master_df["trigger_160"]
            ].groupby("split", observed=False).size(),
            "340 strings": master_df.loc[
                master_df["trigger_340"]
            ].groupby("split", observed=False).size(),
        }
    ).reindex(SPLIT_ORDER)
    geometry_split_fractions = (
        geometry_split_counts / geometry_split_counts.sum(axis=0)
    )

    logger.info(
        "Assigned %s unique physical events.",
        f"{len(master_df):,}",
    )
    logger.info("Event-key duplicates: 0")
    logger.info("Every event has exactly one split: yes")
    logger.info(
        "Every stratum appears in train, validation, and test: yes"
    )
    log_table(logger, "Split summary:", split_summary)
    log_table(logger, "Flavor x split:", flavor_split_counts)
    log_table(logger, "Geometry split counts:", geometry_split_counts)
    log_table(
        logger,
        "Geometry split fractions (%):",
        geometry_split_fractions.mul(100).round(4),
    )
    return master_df


def make_manifest(master_df: pd.DataFrame) -> pl.DataFrame:
    missing_columns = set(MANIFEST_COLUMNS).difference(master_df.columns)
    if missing_columns:
        raise KeyError(
            f"master_df is missing manifest columns: "
            f"{sorted(missing_columns)}"
        )

    manifest_df = master_df[MANIFEST_COLUMNS].copy()
    manifest_df["flavor"] = manifest_df["flavor"].astype(str)
    manifest_df["split"] = manifest_df["split"].astype(str)
    manifest_df["stratum_id"] = manifest_df["stratum_id"].astype("int32")
    for trigger_column in ("trigger_102", "trigger_160", "trigger_340"):
        manifest_df[trigger_column] = manifest_df[trigger_column].astype(bool)

    manifest_df = manifest_df.sort_values(
        EVENT_KEY_COLUMNS,
        kind="mergesort",
    ).reset_index(drop=True)
    if manifest_df.duplicated(EVENT_KEY_COLUMNS).any():
        raise AssertionError(
            "The manifest contains duplicated physical event keys."
        )
    if set(manifest_df["split"].unique()) != set(SPLIT_ORDER):
        raise AssertionError(
            "The manifest does not contain exactly train, validation, and test."
        )

    return pl.DataFrame(
        [
            pl.Series(
                "flavor",
                manifest_df["flavor"].to_numpy(dtype=str),
                dtype=pl.String,
            ),
            pl.Series("RunID", manifest_df["RunID"].to_numpy()),
            pl.Series("SubrunID", manifest_df["SubrunID"].to_numpy()),
            pl.Series("EventID", manifest_df["EventID"].to_numpy()),
            pl.Series("SubEventID", manifest_df["SubEventID"].to_numpy()),
            pl.Series(
                "split",
                manifest_df["split"].to_numpy(dtype=str),
                dtype=pl.String,
            ),
            pl.Series(
                "stratum_id",
                manifest_df["stratum_id"].to_numpy(),
                dtype=pl.Int32,
            ),
            pl.Series(
                "trigger_102",
                manifest_df["trigger_102"].to_numpy(),
                dtype=pl.Boolean,
            ),
            pl.Series(
                "trigger_160",
                manifest_df["trigger_160"].to_numpy(),
                dtype=pl.Boolean,
            ),
            pl.Series(
                "trigger_340",
                manifest_df["trigger_340"].to_numpy(),
                dtype=pl.Boolean,
            ),
        ]
    ).select(MANIFEST_COLUMNS)


def write_manifest(
    manifest: pl.DataFrame,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to replace an existing timestamped manifest: {output_path}"
        )

    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )
    try:
        manifest.write_parquet(temporary_path, compression="zstd")
        written = pl.read_parquet(temporary_path).select(MANIFEST_COLUMNS)
        if not written.equals(manifest):
            raise AssertionError(
                "Temporary parquet differs from the in-memory manifest."
            )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    saved = pl.read_parquet(output_path).select(MANIFEST_COLUMNS)
    if saved.height != manifest.height:
        raise AssertionError(
            "Saved manifest row count does not match the generated manifest."
        )
    if (
        saved.select(pl.struct(EVENT_KEY_COLUMNS).n_unique()).item()
        != saved.height
    ):
        raise AssertionError(
            "Saved manifest contains duplicated physical event keys."
        )

    checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
    logger.info("Saved manifest: %s", output_path)
    logger.info("Rows: %s", f"{saved.height:,}")
    logger.info(
        "Size: %.2f MiB",
        output_path.stat().st_size / (1024**2),
    )
    logger.info("SHA256: %s", checksum)
    logger.info(
        "Saved split counts:\n%s",
        saved.group_by("split").len().sort("split"),
    )


def run(
    args: argparse.Namespace,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    if args.energy_max <= 0:
        raise ValueError("--energy-max must be positive.")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive.")
    if args.minimum_stratum_size <= 0:
        raise ValueError("--minimum-stratum-size must be positive.")
    if args.expected_strata <= 0:
        raise ValueError("--expected-strata must be positive.")

    logger.info("=== MASTER SPLIT BUILD ===")
    logger.info("Truth directory: %s", args.truth_dir)
    logger.info("Output parquet: %s", output_path)
    logger.info("Energy maximum: %g GeV (exclusive)", args.energy_max)
    logger.info("CSV chunk size: %s", f"{args.chunksize:,}")
    logger.info("Expected candidate strata: %s", args.expected_strata)
    logger.info(
        "Expected energy edges: %s",
        args.expected_energy_edges,
    )
    logger.info("Minimum stratum size: %s", args.minimum_stratum_size)
    logger.info("Hash key: %s", args.hash_key)
    master_df = load_selected_truth(
        truth_dir=args.truth_dir.resolve(),
        energy_max=args.energy_max,
        chunksize=args.chunksize,
        logger=logger,
    )
    master_df, _, _ = build_candidate_strata(
        master_df=master_df,
        minimum_size=args.minimum_stratum_size,
        expected_strata=args.expected_strata,
        expected_energy_edges=args.expected_energy_edges,
        logger=logger,
    )
    master_df = assign_splits(
        master_df=master_df,
        hash_key=args.hash_key,
        logger=logger,
    )
    manifest = make_manifest(master_df)
    write_manifest(
        manifest=manifest,
        output_path=output_path,
        logger=logger,
    )
    logger.info("=== DONE ===")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    output_path = output_dir / (
        f"{OUTPUT_BASENAME}_{timestamp}.parquet"
    )
    log_path = output_dir / f"{OUTPUT_BASENAME}_{timestamp}.log"
    if output_path.exists() or log_path.exists():
        print(
            "ERROR: timestamped output already exists for this minute: "
            f"{output_path} or {log_path}",
            file=sys.stderr,
        )
        return 2
    logger = setup_logger(log_path)
    started = time.time()
    try:
        run(args, output_path, logger)
    except Exception:
        logger.exception("MASTER SPLIT BUILD FAILED")
        logger.info("Elapsed seconds: %.1f", time.time() - started)
        return 1
    logger.info("Log file: %s", log_path)
    logger.info("Elapsed seconds: %.1f", time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
