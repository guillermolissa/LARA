"""
Generate comparative bar charts from outputs/metrics.csv.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / "outputs" / "metrics.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "figures"

# Non-metric columns present in the raw CSV.
ID_COLUMNS = ["source", "file_name", "datetime"]

# Canonical ordering for the well-known baseline algorithms so bar order/colors
# stay consistent across datasets. Anything not listed here (i.e. LARA
# variants) is appended afterwards, sorted alphabetically.
BASE_ALGO_ORDER = ["pop", "ar", "sknn", "markov", "gru4rec-l10"]

METRIC_COL_RE = re.compile(r"^(?P<family>[A-Za-z]+)@(?P<k>\d+)$")

# Hyperparameter tokens used to build a short disambiguating suffix for
# datasets with more than one LARA submission (currently only Dressipi).
HYPERPARAM_RE = re.compile(
    r"emb(?P<emb>\d+).*?h(?P<h>\d+).*?l(?P<l>\d+).*?"
    r"dr(?P<dr>[\d.]+).*?ep(?P<ep>\d+)"
)


def normalize_algorithm_name(model: str, source: str) -> str:
    """Map a raw `Model` value to a clean, cross-dataset-comparable label."""
    cleaned = model
    # Drop the dataset name and generic submission boilerplate.
    cleaned = re.sub(re.escape(source), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(^|-)submit(-|$)", "-", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip("-_")

    if "lara" in cleaned.lower():
        return "LARA"  # disambiguated later if a dataset has several variants

    return cleaned.strip("-_")


def disambiguate_lara_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Append a short hyperparameter suffix to LARA rows when a dataset has
    more than one distinct LARA submission, so bars stay distinguishable."""
    df = df.copy()
    for source, group in df.groupby("source"):
        lara_mask = (df["source"] == source) & (df["Algorithm"] == "LARA")
        n_variants = df.loc[lara_mask, "file_name"].nunique()
        if n_variants <= 1:
            continue
        for model in df.loc[lara_mask, "file_name"].unique():
            match = HYPERPARAM_RE.search(model)
            if match:
                p = match.groupdict()
                suffix = f"LARA (emb{p['emb']}, h{p['h']}, l{p['l']}, dr{p['dr']}, ep{p['ep']})"
            else:
                suffix = f"LARA ({model})"
            df.loc[lara_mask & (df["file_name"] == model), "Algorithm"] = suffix
    return df


def load_long_format(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and reshape it to long format: one row per
    (Source, Algorithm, metric family, k, value)."""
    df = pd.read_csv(csv_path)

    metric_columns = [c for c in df.columns if METRIC_COL_RE.match(c)]
    if not metric_columns:
        raise ValueError("No metric columns matching '<NAME>@<K>' were found.")

    df["Algorithm"] = df.apply(
        lambda row: normalize_algorithm_name(row["file_name"], row["source"]), axis=1
    )
    df = disambiguate_lara_labels(df)

    long_df = df.melt(
        id_vars=["source", "Algorithm"],
        value_vars=metric_columns,
        var_name="metric_col",
        value_name="value",
    )
    parsed = long_df["metric_col"].str.extract(METRIC_COL_RE)
    long_df["metric_family"] = parsed["family"].str.upper()
    long_df["k"] = parsed["k"].astype(int)

    # Drop rows with missing metric values so seaborn doesn't render empty bars.
    long_df = long_df.dropna(subset=["value"])

    return long_df.drop(columns="metric_col")


def algorithm_order_and_palette(algorithms: list[str]) -> tuple[list[str], dict]:
    """Build a consistent left-to-right order and color mapping: known
    baselines first (fixed order/colors), LARA variants last (red shades)."""
    baselines = [a for a in BASE_ALGO_ORDER if a in algorithms]
    lara_variants = sorted(a for a in algorithms if a not in BASE_ALGO_ORDER)
    order = baselines + lara_variants

    palette = {}
    base_colors = sns.color_palette("colorblind", n_colors=max(len(baselines), 1))
    for name, color in zip(baselines, base_colors):
        palette[name] = color

    lara_colors = sns.color_palette("Reds", n_colors=max(len(lara_variants), 1) + 1)[1:]
    for name, color in zip(lara_variants, lara_colors):
        palette[name] = color

    return order, palette


def build_metric_figure(
    data: pd.DataFrame, source: str, metric_family: str, algo_order: list[str], palette: dict
) -> Figure:
    """Draw one grouped bar chart (k on x-axis, one bar group per algorithm)
    for a single dataset/metric-family combination. Does not save or close it."""
    subset = data[(data["source"] == source) & (data["metric_family"] == metric_family)]
    present_order = [a for a in algo_order if a in subset["Algorithm"].unique()]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    sns.barplot(
        data=subset,
        x="k",
        y="value",
        hue="Algorithm",
        hue_order=present_order,
        palette=palette,
        order=sorted(subset["k"].unique()),
        ax=ax,
    )

    ax.set_title(f"{metric_family}@k — {source}", fontsize=13, fontweight="bold")
    ax.set_xlabel("k")
    ax.set_ylabel(metric_family)
    ax.legend(
        title="Algorithm",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def save_png(fig: Figure, source: str, metric_family: str) -> Path:
    """Save a figure as a standalone PNG under OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{source.lower()}_{metric_family.lower()}_at_k.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def save_source_pdf(figs: list[Figure], source: str) -> Path:
    """Bundle all figures for a single dataset into one multi-page PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{source.lower()}_metrics_report.pdf"
    with PdfPages(out_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip bundling each dataset's charts into a combined PDF report.",
    )
    args = parser.parse_args()

    long_df = load_long_format(INPUT_CSV)
    algo_order, palette = algorithm_order_and_palette(sorted(long_df["Algorithm"].unique()))
    metric_families = sorted(long_df["metric_family"].unique())

    saved_pngs = []
    saved_pdfs = []
    for source in sorted(long_df["source"].unique()):
        figs = [
            build_metric_figure(long_df, source, metric_family, algo_order, palette)
            for metric_family in metric_families
        ]
        for fig, metric_family in zip(figs, metric_families):
            saved_pngs.append(save_png(fig, source, metric_family))

        if not args.no_pdf:
            saved_pdfs.append(save_source_pdf(figs, source))

        for fig in figs:
            plt.close(fig)

    print(f"Saved {len(saved_pngs)} figures to {OUTPUT_DIR}")
    for path in saved_pngs:
        print(f"  {path.relative_to(REPO_ROOT)}")

    if saved_pdfs:
        print(f"Saved {len(saved_pdfs)} combined PDF reports to {OUTPUT_DIR}")
        for path in saved_pdfs:
            print(f"  {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
