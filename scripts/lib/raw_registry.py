"""Registry for raw electrophysiology recordings and lab logs."""

from dataclasses import dataclass
from pathlib import Path
from .paths import RAW


@dataclass(frozen=True)
class Recording:
    cohort: str
    group: str
    animal: str
    mat_file: Path
    log_file: Path | None
    spl_variants: tuple[str, ...] = ("60dB",)


RECORDINGS: tuple[Recording, ...] = (
    Recording("HHL", "NE", "1", RAW / "hhl/ne/Jan05-2015_ProcessedData_SPL_FLT_IsoDist.mat", RAW / "hhl/ne/logs/Dec03-Jan05-2014-G-Recovery-Acute-0001.docx", ("60dB", "75dB")),
    Recording("HHL", "NE", "2", RAW / "hhl/ne/Jan07-2015_ProcessedData_SPL_FLT_IsoDist.mat", RAW / "hhl/ne/logs/Dec03-Jan07-2014-G-Recovery-Acute-0002.docx", ("60dB", "75dB")),
    Recording("HHL", "NE", "3", RAW / "hhl/ne/Jul01-2015_ProcessedData_SPL_FLT_IsoDist.mat", RAW / "hhl/ne/logs/Jun01-Jul01-2014-G-Recovery-Acute-0003.docx", ("60dB", "75dB")),
    Recording("HHL", "NE", "4", RAW / "hhl/ne/Jul06-2015_ProcessedData_SPL_FLT_IsoDist.mat", RAW / "hhl/ne/logs/Jun01-Jul06-2014-G-Recovery-Acute-0004.docx", ("60dB", "75dB")),
    Recording("HHL", "SH", "1", RAW / "hhl/sh/Jul02-2014_ProcessedData.mat", RAW / "hhl/sh/logs/Jul02-2014-G-0116.docx", ("75dB",)),
    Recording("HHL", "SH", "2", RAW / "hhl/sh/Aug20-2014_ProcessedData.mat", RAW / "hhl/sh/logs/Aug20-2014-G-0136.docx", ("75dB",)),
    Recording("HHL", "SH", "3", RAW / "hhl/sh/Sep17-2014_ProcessedData.mat", RAW / "hhl/sh/logs/Sep17-2014-G-0138.docx", ("60dB", "75dB")),
    Recording("HHL", "SH", "4", RAW / "hhl/sh/Mar25-2015_ProcessedData.mat", RAW / "hhl/sh/logs/Mar25-2015-G-0146.docx", ("60dB", "75dB")),
    Recording("HHL", "SH", "5", RAW / "hhl/sh/Apr22-2016_ProcessedData.mat", RAW / "hhl/sh/logs/Apr22-2016-G-0151.docx", ("60dB", "75dB")),
    Recording("HHL", "SH", "6a", RAW / "hhl/sh/Dec12-2017_SPL_ProcessedData_Penetration_1.mat", RAW / "hhl/sh/logs/Dec12-2017-G-00028.docx", ("orig_labels",)),
    Recording("HHL", "SH", "6b", RAW / "hhl/sh/Dec12-2017_SPL_ProcessedData_Penetration_2.mat", RAW / "hhl/sh/logs/Dec12-2017-G-00028.docx", ("orig_labels",)),
    Recording("CHL", "EP", "1", RAW / "chl/ep/Jul03-2017_ProcessedData_EarPlug_ProcessedData_CLC_IsoDist.mat", RAW / "chl/logs/Jun28-Jul03-2017-G-Recovery-Acute-0018_Ver_02.docx"),
    Recording("CHL", "PP", "1", RAW / "chl/pp/Jul03-2017_ProcessedData_PostPlug_ProcessedData_CLC_IsoDist.mat", RAW / "chl/logs/Jun28-Jul03-2017-G-Recovery-Acute-0018_Ver_02.docx"),
    Recording("CHL", "EP", "2", RAW / "chl/ep/Jul05-2017_ProcessedData_EarPlug_ProcessedData_CLC_IsoDist.mat", RAW / "chl/logs/Jun28-Jul05-2017-G-Recovery-Acute-0019.docx"),
    Recording("CHL", "PP", "2", RAW / "chl/pp/Jul05-2017_ProcessedData_PostPlug_ProcessedData_CLC_IsoDist.mat", RAW / "chl/logs/Jun28-Jul05-2017-G-Recovery-Acute-0019.docx"),
    Recording("CHL", "EP", "3", RAW / "chl/ep/Oct09-2017_ProcessedData_EarPlug.mat", RAW / "chl/logs/Oct02-Oct09-2017-G-Recovery-Acute-0020.docx"),
    Recording("CHL", "PP", "3", RAW / "chl/pp/Oct09-2017_ProcessedData_PostPlug.mat", RAW / "chl/logs/Oct02-Oct09-2017-G-Recovery-Acute-0020.docx"),
    Recording("CHL", "EP", "4", RAW / "chl/ep/Oct11-2017_ProcessedData_EarPlug.mat", RAW / "chl/logs/Oct02-Oct11-2017-G-Recovery-Acute-0021.docx"),
    Recording("CHL", "PP", "4", RAW / "chl/pp/Oct11-2017_ProcessedData_PostPlug.mat", RAW / "chl/logs/Oct02-Oct11-2017-G-Recovery-Acute-0021.docx"),
)


def recordings(cohort: str | None = None) -> tuple[Recording, ...]:
    if cohort is None:
        return RECORDINGS
    return tuple(r for r in RECORDINGS if r.cohort == cohort)


def validate_registry(require_files: bool = True) -> list[str]:
    problems: list[str] = []
    for rec in RECORDINGS:
        if require_files and not rec.mat_file.exists():
            problems.append(f"missing mat file: {rec.mat_file}")
        if require_files and rec.log_file is not None and not rec.log_file.exists():
            problems.append(f"missing log file: {rec.log_file}")
    # CHL is paired: each animal must have one EP and one PP recording.
    for animal in {r.animal for r in RECORDINGS if r.cohort == "CHL"}:
        groups = sorted(r.group for r in RECORDINGS if r.cohort == "CHL" and r.animal == animal)
        if groups != ["EP", "PP"]:
            problems.append(f"CHL animal {animal} is not paired EP/PP: {groups}")
    return problems
