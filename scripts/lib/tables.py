import pandas as pd
from .paths import DERIVED


def empirical_tests() -> pd.DataFrame:
    return pd.read_csv(DERIVED / "tables" / "empirical_tests.csv")


def bootstrap_contrasts() -> pd.DataFrame:
    return pd.read_csv(DERIVED / "tables" / "bootstrap_contrasts.csv")


def main_text_summary() -> dict:
    empirical = empirical_tests().set_index("id")
    bootstrap = bootstrap_contrasts().set_index("id")
    return {
        "E1_gain_context_p_fdr": float(empirical.loc["E1", "p_BH"]),
        "E8_ep_vs_sh_threshold_p_fdr": float(empirical.loc["E8", "p_BH"]),
        "HE1_low_threshold_pr_dir": float(bootstrap.loc["HE1", "pr_dir"]),
        "HD1_all_units_pr_dir": float(bootstrap.loc["HD1", "pr_dir"]),
    }
