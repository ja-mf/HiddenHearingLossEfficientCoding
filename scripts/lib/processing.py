"""Raw-data processing entry points for the analysis code."""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

from .paths import DATA, RAW
from .raw_registry import Recording, recordings

INTENSITY_LEVELS_DB = np.arange(24, 97, 3, dtype=float)
HPR_CENTERS_DB = [30, 45, 60, 75, 90]
UNIFORM_HPR_CODE = -1
COND_TO_HPR = {0: 90, 1: 75, 2: 60, 3: 45, 4: 30, 5: -1}
DT_SECONDS = 1920 / 48828.125
N_BINS = 2000

SPL_FIELDS = {
    "60dB": {0: "spl_60dB_h01", 1: "spl_60dB_h06", 2: "spl_60dB_h11", 3: "spl_60dB_h16", 4: "spl_60dB_h21", 5: "spl_60dB_flt"},
    "75dB": {0: "spl_75dB_h01", 1: "spl_75dB_h06", 2: "spl_75dB_h11", 3: "spl_75dB_h16", 4: "spl_75dB_h21", 5: "spl_75dB_flt"},
}


def spike_times_to_binary(spike_times: np.ndarray) -> np.ndarray:
    binary = np.zeros(N_BINS, dtype=np.int16)
    if spike_times is None:
        return binary
    idx = np.floor(np.asarray(spike_times, dtype=float) / DT_SECONDS).astype(int)
    idx = idx[(idx >= 0) & (idx < N_BINS)]
    binary[idx] = 1
    return binary


def load_stimulus_sequences() -> dict[int, np.ndarray]:
    """Load the exact stimulus-level sequences used for the recordings."""
    path = RAW / "stimulus_sequences.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing stimulus sequence file: {path}")
    seq = pd.read_csv(path)
    return {int(cond): grp.sort_values("bin")["lvl"].to_numpy(dtype=float)[:N_BINS]
            for cond, grp in seq.groupby("condition")}


def stimulus_sequence_for_condition(cond: int) -> np.ndarray:
    return load_stimulus_sequences()[cond]


def compute_rlf(binary_spikes: np.ndarray, stimulus_levels: np.ndarray) -> dict[float, float]:
    rows: dict[float, float] = {}
    for lvl in INTENSITY_LEVELS_DB:
        mask = stimulus_levels == lvl
        if mask.any():
            rows[float(lvl)] = float(binary_spikes[mask].sum() * 1000.0 / (mask.sum() * 40.0))
    return rows


_FREQ_COND = {"h01": 0, "h06": 1, "h11": 2, "h16": 3, "h21": 4, "flt": 5}


def _orig_label_fields(unit) -> list[tuple[str, int]]:
    """Some recordings expose SPL series under per-frequency labels
    (an artifact of an earlier protocol that shared the same stimulus). 
    Processing them as plain SPL series: pick the first frequency that has at least 
    five HPR conditions with spike data."""
    attrs = [a for a in dir(unit) if a.startswith("spl_") and "Hz_" in a]
    grouped: dict[str, list[tuple[str, int]]] = {}
    for name in attrs:
        parts = name.split("_")
        if len(parts) < 3:
            continue
        freq, cond_str = parts[1], "_".join(parts[2:])
        if cond_str in _FREQ_COND:
            grouped.setdefault(freq, []).append((name, _FREQ_COND[cond_str]))
    for freq in sorted(grouped):
        valid = [(n, c) for n, c in grouped[freq]
                 if getattr(unit, n, None) is not None and hasattr(getattr(unit, n), "sts")]
        if len(valid) >= 5:
            return valid
    if grouped:
        best = max(grouped, key=lambda f: len(grouped[f]))
        return [(n, c) for n, c in grouped[best]
                if getattr(unit, n, None) is not None and hasattr(getattr(unit, n), "sts")]
    return []


def available_fields(unit, variant: str) -> list[tuple[str, int]]:
    if variant == "orig_labels":
        return _orig_label_fields(unit)
    fields = []
    for cond, field in SPL_FIELDS.get(variant, {}).items():
        if hasattr(unit, field):
            fields.append((field, cond))
    return fields


def extract_spike_times(unit, field: str):
    obj = getattr(unit, field)
    if hasattr(obj, "sts"):
        return obj.sts
    if isinstance(obj, np.ndarray) and obj.size and hasattr(obj.flat[0], "sts"):
        return obj.flat[0].sts
    return None


def process_recording(rec: Recording) -> pd.DataFrame:
    data = sio.loadmat(rec.mat_file, squeeze_me=True, struct_as_record=False)
    units = data["data"]
    if not hasattr(units, "__len__"):
        units = np.array([units])
    rows = []
    stim = load_stimulus_sequences()
    for variant in rec.spl_variants:
        for unit_index, unit in enumerate(units):
            neuron_id = f"{rec.group}{rec.animal}u{unit_index}"
            for field, cond in available_fields(unit, variant):
                if cond not in stim:
                    continue
                spikes = spike_times_to_binary(extract_spike_times(unit, field))
                rlf = compute_rlf(spikes, stim[cond])
                for lvl, spks in rlf.items():
                    rows.append({
                        "group": rec.group,
                        "animal_id": rec.animal,
                        "neuron_id": neuron_id,
                        "spl_variant": variant,
                        "hpr": COND_TO_HPR[cond],
                        "lvl": lvl,
                        "spks": spks,
                    })
    return pd.DataFrame(rows)


def build_rlfs(cohort: str, output: Path | None = None) -> pd.DataFrame:
    frames = [process_recording(rec) for rec in recordings(cohort)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if output is None:
        output = DATA / "artifacts" / f"rlfs_{cohort}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    return df


def describe_pipeline() -> list[str]:
    return [
        "read raw MATLAB single-unit recordings",
        "compute rate-level functions on the 24-96 dB SPL grid",
        "fit normalized sigmoid threshold/gain parameters on the analysis grid",
        "apply the quality filter (max mean MSE <= 0.08)",
        "compute empirical mixed-effects tests and optimization-prior summaries",
    ]
