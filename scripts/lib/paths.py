from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
DERIVED = DATA / "derived"
FIGURES = ROOT / "figures"
CONFIG = ROOT / "config" / "reproduction.yaml"
