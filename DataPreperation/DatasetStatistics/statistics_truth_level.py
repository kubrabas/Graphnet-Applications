"""Write truth-level event statistics for P-ONE PMT-response files."""
# example usage: sbatch --cpus-per-task=100 /home/kbas/SlurmScripts/DataPreperation/submit_truth_level_statistics.sh NC 100

import argparse
import csv
import importlib.util
import math
import os
import re
import shutil
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py  # noqa: F401  # import before icecube/graphnet to avoid HDF5 conflicts
from icecube import LeptonInjector, dataclasses, dataio, icetray  # noqa: F401
import LeptonWeighter as LW


GRAPHNET_SRC = Path("/project/def-nahee/kbas/graphnet/src")
if str(GRAPHNET_SRC) not in sys.path:
    sys.path.insert(0, str(GRAPHNET_SRC))

from graphnet.data.extractors.icecube.i3truthextractor import (  # noqa: E402
    I3TruthExtractorPONE,
)


PATHS_PY = Path("/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py")
DEFAULT_OUTDIR = Path(
    "/project/def-nahee/kbas/Graphnet-Applications/Metadata/"
    "DatasetStatistics/TruthLevelStatistics"
)
BERLIN_TZ = ZoneInfo("Europe/Berlin")
XS_PATH = Path(
    "/project/6008051/pone_simulation/pone_offline/"
    "CrossSectionModels/csms_differential_v1.0"
)
DEFAULT_NUSQUIDS_WEIGHT_TABLE = Path(
    "/cvmfs/software.pacific-neutrino.org/pone_offline/v2.0/"
    "data/nsq_allneu_propagation_weight_gamma2.h5"
)
N_GENERATED_PER_PARTICLE = 100
EXPECTED_EVENTS_PER_NORMAL_BATCH = 2 * N_GENERATED_PER_PARTICLE
LIC_SIZE_T_BYTES = 8

LW_PARTICLE_FROM_PDG = {
    12: LW.NuE, -12: LW.NuEBar,
    14: LW.NuMu, -14: LW.NuMuBar,
    16: LW.NuTau, -16: LW.NuTauBar,
    11: LW.EMinus, -11: LW.EPlus,
    13: LW.MuMinus, -13: LW.MuPlus,
    15: LW.TauMinus, -15: LW.TauPlus,
    -2000001006: LW.Hadrons,
}

_xs = None
_survival = None
_extractor = None

COLUMNS = [
    "RunID",
    "SubrunID",
    "EventID",
    "SubEventID",
    "position_x",
    "position_y",
    "position_z",
    "pid",
    "is_CC",
    "totalEnergy",
    "zenith",
    "azimuth",
    "finalStateX",
    "finalStateY",
    "finalType1",
    "finalType2",
    "initialType",
    "totalColumnDepth",
    "impactParameter",
    "triggered_noisy_340_string",
    "trigger_time_noisy_340_string",
    "triggered_noisy_160_string",
    "trigger_time_noisy_160_string",
    "triggered_noisy_102_string",
    "trigger_time_noisy_102_string",
    "triggered_nonoise_340_string",
    "trigger_time_nonoise_340_string",
    "triggered_nonoise_160_string",
    "trigger_time_nonoise_160_string",
    "triggered_nonoise_102_string",
    "trigger_time_nonoise_102_string",
]

WEIGHT_COLUMNS = ["oneweight", "survivalProb", "final_weight"]
CHUNK_COLUMNS = COLUMNS + ["_primary_pdg", "_raw_oneweight", "survivalProb"]
OUTPUT_COLUMNS = COLUMNS + WEIGHT_COLUMNS


def load_paths():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract_batch_id(path: Path):
    match = re.search(r"(\d+)(?:\.[^.]+)+$", path.name)
    return str(int(match.group(1))) if match else None


def build_unique_batch_map(directory: Path, pattern: str):
    result = {}
    duplicates = {}
    for path in discover_inputs(directory, pattern):
        batch_id = extract_batch_id(path)
        if batch_id is None:
            raise RuntimeError(f"Could not extract batch ID from {path}")
        if batch_id in result:
            duplicates.setdefault(batch_id, [result[batch_id]]).append(path)
        else:
            result[batch_id] = path
    if duplicates:
        details = "; ".join(
            f"{bid}: {', '.join(map(str, files))}"
            for bid, files in sorted(duplicates.items())
        )
        raise RuntimeError(f"Duplicate batch IDs found: {details}")
    return result


def to_lw_particle(particle_type):
    pdg = int(particle_type)
    if pdg not in LW_PARTICLE_FROM_PDG:
        raise ValueError(f"Unsupported LeptonWeighter particle type: {pdg}")
    return LW_PARTICLE_FROM_PDG[pdg]


def load_xs():
    return LW.CrossSectionFromSpline(
        str(XS_PATH / "dsdxdy_nu_CC_iso.fits"),
        str(XS_PATH / "dsdxdy_nubar_CC_iso.fits"),
        str(XS_PATH / "dsdxdy_nu_NC_iso.fits"),
        str(XS_PATH / "dsdxdy_nubar_NC_iso.fits"),
    )


class SurvivalProbability:
    def __init__(self, table_path):
        import nuSQuIDS as nsq

        self.nsq_atm = nsq.nuSQUIDSAtm(str(table_path))
        self.units = nsq.Const()
        cosrange = list(self.nsq_atm.GetCosthRange())
        energyrange = list(self.nsq_atm.GetERange())
        self.cosmin, self.cosmax = min(cosrange), max(cosrange)
        self.emin, self.emax = min(energyrange), max(energyrange)

    @staticmethod
    def _indices(primary_type):
        flavor = 1 if primary_type in (LW.NuMu, LW.NuMuBar) else 0
        if primary_type in (LW.NuTau, LW.NuTauBar):
            flavor = 2
        anti = int(primary_type in (LW.NuEBar, LW.NuMuBar, LW.NuTauBar))
        return flavor, anti

    @staticmethod
    def _eflux(energy):
        return 1e18 * energy ** -2.0

    def __call__(self, energy_gev, zenith, primary_type):
        energy = energy_gev * self.units.GeV
        cos_zenith = math.cos(zenith)
        flavor, anti = self._indices(primary_type)
        if self.cosmin < cos_zenith < self.cosmax:
            if self.emin < energy < self.emax:
                return self.nsq_atm.EvalFlavor(flavor, cos_zenith, energy, anti) / self._eflux(energy)
            return 1.0 if energy <= self.emin else 0.0
        if self.emin < energy < self.emax:
            return self.nsq_atm.EvalFlavor(flavor, 0.0, energy, anti) / self._eflux(energy)
        return 0.0


def read_lic_injection_configs(lic_path):
    configs = []
    data = Path(lic_path).read_bytes()
    offset = 0
    while offset + 17 <= len(data):
        block_start = offset
        try:
            block_len = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            name_len = struct.unpack_from("<Q", data, offset)[0]
            offset += LIC_SIZE_T_BYTES
            name = data[offset:offset + name_len].decode("utf-8")
            offset += name_len + 1
        except (struct.error, UnicodeDecodeError):
            break
        payload_len = block_len - (17 + name_len)
        payload_end = offset + payload_len
        if block_len < 17 + name_len or payload_end > len(data):
            break
        payload = data[offset:payload_end]
        if name in ("RangedInjectionConfiguration", "VolumeInjectionConfiguration"):
            if len(payload) < 16:
                raise RuntimeError(f"Short injection configuration block in {lic_path}")
            mode = "range" if name == "RangedInjectionConfiguration" else "volume"
            radius = struct.unpack_from("<d", payload, len(payload) - 16)[0]
            configs.append((mode, radius))
        offset = block_start + block_len
    return configs


def require_lic_injection_config(lic_path):
    """Return the single injection configuration encoded in the LIC file."""
    unique = set(read_lic_injection_configs(lic_path))
    if len(unique) != 1:
        raise RuntimeError(
            f"Expected exactly one unique LIC injection mode/radius in {lic_path}; "
            f"found {sorted(unique)}"
        )
    mode, radius = next(iter(unique))
    if not math.isfinite(radius) or radius <= 0:
        raise RuntimeError(f"Invalid injection radius {radius} in {lic_path}")

    # Old LIW logic used a three-step fallback: frame LeptonInjectorProperties,
    # then the matched LIC binary, then a hard-coded 500 m default. The merged
    # truth/weight pipeline intentionally accepts only the matched LIC value.
    # The removed fallback was equivalent to:
    #   if frame.Has("LeptonInjectorProperties"):
    #       radius = props.injectionRadius or props.cylinderRadius
    #   elif LIC parsing succeeded:
    #       radius = lic_radius
    #   else:
    #       radius = 500.0
    return mode, radius


def worker_init(gcd_file, nusquids_table):
    global _xs, _survival, _extractor
    _xs = load_xs()
    _survival = SurvivalProbability(nusquids_table)
    _extractor = I3TruthExtractorPONE(name="truth", exclude=[])
    _extractor.set_gcd(gcd_file)


def scalar(value):
    """Convert IceTray scalar-like objects to CSV-friendly Python values."""
    if hasattr(value, "value"):
        value = value.value

    if isinstance(value, (bool, int, float, str)) or value is None:
        return value

    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def process_file(task):
    """Extract truth and unscaled weights from one PMT-response file."""
    index, batch_id, pmt_s, lic_s, chunk_s, expected_daq = task
    pmt_file = Path(pmt_s)
    lic_file = Path(lic_s)
    chunk = Path(chunk_s)
    t_start = time.time()

    _, injection_radius = require_lic_injection_config(lic_file)
    generators = LW.MakeGeneratorsFromLICFile(str(lic_file))
    weighter = LW.Weighter(_xs, generators)

    rows = 0
    nu_rows = 0
    nubar_rows = 0
    event_ids = set()
    chunk.parent.mkdir(parents=True, exist_ok=True)

    with chunk.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CHUNK_COLUMNS)
        writer.writeheader()
        i3_file = dataio.I3File(str(pmt_file), "r")
        try:
            while i3_file.more():
                try:
                    frame = i3_file.pop_daq()
                except RuntimeError as exc:
                    if "no frame to pop" in str(exc).lower():
                        break
                    raise RuntimeError(f"Failed reading DAQ frame from {pmt_file}: {exc}") from exc

                missing = [key for key in ("I3EventHeader", "EventProperties") if key not in frame]
                if missing:
                    raise RuntimeError(f"Missing required keys {missing} in {pmt_file}, DAQ {rows + 1}")

                header = frame["I3EventHeader"]
                event_id = (
                    int(header.run_id), int(header.sub_run_id),
                    int(header.event_id), int(header.sub_event_id),
                )
                if event_id in event_ids:
                    raise RuntimeError(f"Duplicate event ID {event_id} within {pmt_file}")
                event_ids.add(event_id)

                ep = frame["EventProperties"]
                primary = to_lw_particle(ep.initialType)
                event = LW.Event()
                event.energy = ep.totalEnergy
                event.zenith = ep.zenith
                event.azimuth = ep.azimuth
                event.interaction_x = ep.finalStateX
                event.interaction_y = ep.finalStateY
                event.primary_type = primary
                event.final_state_particle_0 = to_lw_particle(ep.finalType1)
                event.final_state_particle_1 = to_lw_particle(ep.finalType2)
                event.radius = injection_radius
                event.total_column_depth = ep.totalColumnDepth
                event.x, event.y, event.z = ep.x, ep.y, ep.z

                raw_oneweight = float(weighter.get_oneweight(event))
                survival_prob = float(_survival(event.energy, event.zenith, primary))
                if not math.isfinite(raw_oneweight) or raw_oneweight < 0:
                    raise RuntimeError(f"Invalid raw oneweight {raw_oneweight} for event {event_id}")
                # nuSQuIDS propagation weights can be slightly above one due
                # to regeneration, so only finiteness and non-negativity are
                # valid generic checks here.
                if not math.isfinite(survival_prob) or survival_prob < 0.0:
                    raise RuntimeError(f"Invalid survival probability {survival_prob} for event {event_id}")

                extracted = _extractor(frame)
                row = {column: scalar(extracted.get(column, -1)) for column in COLUMNS}
                row["_primary_pdg"] = int(ep.initialType)
                row["_raw_oneweight"] = raw_oneweight
                row["survivalProb"] = survival_prob
                writer.writerow(row)
                rows += 1
                if int(ep.initialType) > 0:
                    nu_rows += 1
                else:
                    nubar_rows += 1
        finally:
            i3_file.close()

    if rows != expected_daq:
        raise RuntimeError(
            f"Batch {batch_id} produced {rows} weighted PMT DAQ rows; expected {expected_daq}"
        )
    return {
        "index": index,
        "batch_id": batch_id,
        "file": str(pmt_file),
        "lic_file": str(lic_file),
        "chunk": str(chunk),
        "status": "ok",
        "rows": rows,
        "nu_rows": nu_rows,
        "nubar_rows": nubar_rows,
        "expected_daq": expected_daq,
        "elapsed": time.time() - t_start,
    }


def discover_inputs(indir: Path, pattern: str):
    return sorted(path for path in indir.rglob(pattern) if path.is_file())


def merge_chunks(results, out: Path, n_nu: int, n_nubar: int):
    """Normalize raw weights while atomically merging worker chunks."""
    if n_nu <= 0 or n_nubar <= 0:
        raise RuntimeError(f"Cannot normalize sample with nu={n_nu}, nubar={n_nubar}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary_out = out.with_name(f".{out.name}.tmp")
    seen_event_ids = set()
    with temporary_out.open("w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for result in sorted(results, key=lambda item: item["index"]):
            with Path(result["chunk"]).open("r", newline="") as in_fh:
                reader = csv.DictReader(in_fh)
                for row in reader:
                    event_id = tuple(row[key] for key in ("RunID", "SubrunID", "EventID", "SubEventID"))
                    if event_id in seen_event_ids:
                        raise RuntimeError(f"Duplicate event ID across PMT files: {event_id}")
                    seen_event_ids.add(event_id)

                    primary_pdg = int(row["_primary_pdg"])
                    denominator = n_nu if primary_pdg > 0 else n_nubar
                    raw_oneweight = float(row["_raw_oneweight"])
                    survival_prob = float(row["survivalProb"])
                    oneweight = raw_oneweight * N_GENERATED_PER_PARTICLE / denominator
                    final_weight = oneweight * survival_prob
                    if not all(map(math.isfinite, (oneweight, final_weight))):
                        raise RuntimeError(f"Non-finite normalized weight for event {event_id}")

                    output_row = {column: row.get(column, -1) for column in COLUMNS}
                    output_row.update({
                        "oneweight": oneweight,
                        "survivalProb": survival_prob,
                        "final_weight": final_weight,
                    })
                    writer.writerow(output_row)
    temporary_out.replace(out)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract selected truth-level statistics from PMT-response I3 files."
    )
    ap.add_argument("--flavor", required=True, help="Flavor key, e.g. Muon, Electron, Tau, NC")
    ap.add_argument("--pmt-dir", required=True, help="Directory containing PMT-response I3 files")
    ap.add_argument("--pmt-pattern", default="*.i3.gz", help="PMT-response glob under --pmt-dir")
    ap.add_argument("--lic-dir", required=True, help="Directory containing matched LIC files")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--logdir", default=None, help="Log/chunk directory")
    ap.add_argument("--nworkers", type=int, default=None, help="Parallel workers")
    ap.add_argument("--mc-key", default="String340MC", help="BAD_I3_FILES top-level key")
    ap.add_argument("--gcd", default=None, help="GCD file for I3TruthExtractorPONE.set_gcd")
    ap.add_argument(
        "--nusquids-weight-table",
        default=str(DEFAULT_NUSQUIDS_WEIGHT_TABLE),
        help="nuSQuIDS propagation table",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths()

    pmt_dir = Path(args.pmt_dir).resolve()
    lic_dir = Path(args.lic_dir).resolve()
    out = Path(args.out).resolve() if args.out else (DEFAULT_OUTDIR / f"{args.flavor}.csv").resolve()
    logdir = Path(args.logdir).resolve() if args.logdir else out.parent
    logdir.mkdir(parents=True, exist_ok=True)

    gcd_file = args.gcd or paths.GCD["340StringMC"]
    pmt_map = build_unique_batch_map(pmt_dir, args.pmt_pattern)
    lic_map = build_unique_batch_map(lic_dir, "*.lic")
    if not pmt_map:
        raise RuntimeError(f"No PMT files found in {pmt_dir} matching {args.pmt_pattern}")
    missing_lic = sorted(set(pmt_map) - set(lic_map), key=int)
    if missing_lic:
        raise RuntimeError(f"PMT batches without matched LIC files: {missing_lic}")

    bad_info = getattr(paths, "BAD_I3_FILES", {}).get(args.mc_key, {}).get(args.flavor, {})
    expected_by_batch = {}
    for upstream_path, available_count in bad_info.get("available_daq_counts", {}).items():
        batch_id = extract_batch_id(Path(upstream_path))
        if batch_id is None:
            raise RuntimeError(f"Could not extract batch ID from BAD_I3_FILES path {upstream_path}")
        if batch_id in expected_by_batch:
            raise RuntimeError(f"Duplicate available_daq_counts batch ID {batch_id}")
        expected_by_batch[batch_id] = int(available_count)

    nworkers = args.nworkers or int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    chunks = logdir / "chunks" / f"{args.flavor}_{job_id}"
    chunks.mkdir(parents=True, exist_ok=True)
    started = datetime.now(BERLIN_TZ)
    summary_log = logdir / f"truth_level_statistics_{args.flavor}_{job_id}.log"

    tasks = [
        (
            index,
            batch_id,
            str(pmt_map[batch_id]),
            str(lic_map[batch_id]),
            str(chunks / f"{index:06d}_{pmt_map[batch_id].stem}.csv"),
            expected_by_batch.get(batch_id, EXPECTED_EVENTS_PER_NORMAL_BATCH),
        )
        for index, batch_id in enumerate(sorted(pmt_map, key=int))
    ]

    results = []
    t_job_start = time.time()
    with summary_log.open("w") as log:
        def log_line(message=""):
            print(message)
            log.write(message + "\n")
            log.flush()

        log_line("=== TRUTH + EVENT WEIGHT JOB ===")
        log_line(f"job_id   : {job_id}")
        log_line(f"started  : {started.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        log_line(f"flavor   : {args.flavor}")
        log_line(f"pmt_dir  : {pmt_dir}")
        log_line(f"pattern  : {args.pmt_pattern}")
        log_line(f"lic_dir  : {lic_dir}")
        log_line(f"gcd      : {gcd_file}")
        log_line(f"out      : {out}")
        log_line(f"logdir   : {logdir}")
        log_line(f"files    : {len(pmt_map)}")
        log_line(f"workers  : {nworkers}")
        log_line()

        with ProcessPoolExecutor(
            max_workers=nworkers,
            initializer=worker_init,
            initargs=(gcd_file, args.nusquids_weight_table),
        ) as executor:
            futures = [executor.submit(process_file, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                log_line(
                    f"{result['status']:3s} batch={result['batch_id']:>6s} "
                    f"rows={result['rows']:4d}/{result['expected_daq']:4d} "
                    f"nu={result['nu_rows']:4d} nubar={result['nubar_rows']:4d} "
                    f"elapsed={result['elapsed']:8.1f}s file={result['file']}"
                )

        n_nu = sum(result["nu_rows"] for result in results)
        n_nubar = sum(result["nubar_rows"] for result in results)
        log_line(f"available neutrinos     : {n_nu}")
        log_line(f"available antineutrinos : {n_nubar}")
        merge_chunks(results, out, n_nu, n_nubar)
        shutil.rmtree(chunks)

        n_success = sum(result["status"] == "ok" for result in results)
        n_rows = sum(result["rows"] for result in results)
        elapsed = time.time() - t_job_start
        log_line()
        log_line("=== DONE ===")
        log_line(f"success_files : {n_success}")
        log_line(f"rows          : {n_rows}")
        log_line(f"elapsed_s     : {elapsed:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
