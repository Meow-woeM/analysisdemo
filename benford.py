#!/usr/bin/env python3
"""
Benford's Law Fraud Detection Script
=====================================
Applies Benford's Law to numeric columns in an Excel file to identify
potential anomalies that may warrant further investigation.

Benford's Law states that in many naturally occurring datasets, the
leading digit d (1-9) occurs with probability P(d) = log10(1 + 1/d).
Significant deviations from this expected distribution can be an
indicator of data manipulation or fraud.

Outputs:
  - output/benford_report.txt       (detailed text report)
  - output/benford_summary.json     (machine-readable summary)
  - output/chart_benford_*.png      (comparison charts per column)
  - output/benford_flagged.csv      (rows with anomalous leading digits)

Usage:
    python benford.py                          # defaults to je_samples.xlsx
    python benford.py path/to/other_file.xlsx  # analyse a different file
"""

import json
import math
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import chi2

# ── Configuration ────────────────────────────────────────────────────────────
DEFAULT_FILE = "je_samples.xlsx"
OUTPUT_DIR = Path("output")
CHART_DPI = 150
sns.set_theme(style="whitegrid", palette="muted")

# Columns to prioritize for Benford analysis (accounting amounts)
PRIORITY_COLUMNS = ["Amount", "Debit", "Credit", "AbsoluteAmount"]

# Chi-squared critical value significance level
SIGNIFICANCE_LEVEL = 0.05
# Degrees of freedom for leading digit test (9 digits - 1)
DF = 8

# Threshold for per-digit deviation to flag as anomalous (percentage points)
DIGIT_DEVIATION_THRESHOLD = 3.0


# ── Benford helpers ──────────────────────────────────────────────────────────
def benford_expected(digit: int) -> float:
    """Return the expected Benford probability for a leading digit (1-9)."""
    return math.log10(1 + 1 / digit)


EXPECTED_PROBS = {d: benford_expected(d) for d in range(1, 10)}


def leading_digit(x) -> int | None:
    """Extract the leading (first non-zero) digit from a number."""
    try:
        x = abs(float(x))
    except (ValueError, TypeError):
        return None
    if x == 0 or not math.isfinite(x):
        return None
    while x < 1:
        x *= 10
    return int(str(x)[0])


def compute_benford(series: pd.Series) -> dict:
    """
    Compute Benford analysis on a numeric series.

    Returns a dict with observed/expected distributions, chi-squared
    statistic, p-value, and per-digit deviations.
    """
    digits = series.apply(leading_digit).dropna().astype(int)
    n = len(digits)
    if n < 50:
        return None

    observed_counts = digits.value_counts().reindex(range(1, 10), fill_value=0)
    observed_pct = (observed_counts / n) * 100
    expected_pct = pd.Series({d: EXPECTED_PROBS[d] * 100 for d in range(1, 10)})
    expected_counts = pd.Series({d: EXPECTED_PROBS[d] * n for d in range(1, 10)})

    # Chi-squared statistic
    chi2_stat = ((observed_counts - expected_counts) ** 2 / expected_counts).sum()
    p_value = 1 - chi2.cdf(chi2_stat, DF)

    # Per-digit deviation (percentage points)
    deviation = observed_pct - expected_pct

    # Flag digits with large deviations
    flagged_digits = [
        int(d) for d in range(1, 10)
        if abs(deviation[d]) > DIGIT_DEVIATION_THRESHOLD
    ]

    return {
        "n": n,
        "observed_pct": observed_pct.to_dict(),
        "expected_pct": expected_pct.to_dict(),
        "deviation": deviation.to_dict(),
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < SIGNIFICANCE_LEVEL),
        "flagged_digits": flagged_digits,
    }


# ── Reporting ────────────────────────────────────────────────────────────────
def section(title: str) -> str:
    line = "=" * 70
    return f"\n{line}\n  {title}\n{line}\n"


def build_report(results: dict, filepath: str) -> str:
    """Build a text report from the Benford analysis results."""
    lines: list[str] = []

    lines.append(section("BENFORD'S LAW FRAUD DETECTION REPORT"))
    lines.append(f"  Source file     : {filepath}")
    lines.append(f"  Analysis run at : {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"  Significance    : {SIGNIFICANCE_LEVEL} (alpha)")
    lines.append(f"  Digit deviation : >{DIGIT_DEVIATION_THRESHOLD} pp flagged")

    lines.append(section("WHAT IS BENFORD'S LAW?"))
    lines.append("  Benford's Law predicts the frequency of leading digits in many")
    lines.append("  real-world numeric datasets. The expected distribution is:")
    lines.append("")
    lines.append("    Digit | Expected %")
    lines.append("    ------+-----------")
    for d in range(1, 10):
        lines.append(f"      {d}   | {EXPECTED_PROBS[d]*100:5.1f}%")
    lines.append("")
    lines.append("  Deviations from this pattern may indicate fabricated or")
    lines.append("  manipulated data, though they can also have innocent causes.")

    for col, result in results.items():
        lines.append(section(f"ANALYSIS: {col}"))

        if result is None:
            lines.append("  Skipped: fewer than 50 usable values.")
            continue

        lines.append(f"  Usable values (n)  : {result['n']:,}")
        lines.append(f"  Chi-squared stat   : {result['chi2_stat']:.2f}")
        lines.append(f"  p-value            : {result['p_value']:.6f}")
        verdict = "YES - distribution deviates significantly" if result["significant"] \
            else "No - distribution is consistent with Benford's Law"
        lines.append(f"  Statistically sig. : {verdict}")
        lines.append("")

        lines.append(f"  {'Digit':<7s} {'Observed%':>10s} {'Expected%':>10s} {'Deviation':>10s} {'Flag':>6s}")
        lines.append("  " + "-" * 47)
        for d in range(1, 10):
            obs = result["observed_pct"][d]
            exp = result["expected_pct"][d]
            dev = result["deviation"][d]
            flag = " ***" if d in result["flagged_digits"] else ""
            lines.append(f"    {d:<5d} {obs:>10.2f} {exp:>10.2f} {dev:>+10.2f}{flag}")

        if result["flagged_digits"]:
            lines.append(f"\n  Anomalous digits: {result['flagged_digits']}")
            lines.append("  These digits appear more or less frequently than Benford's")
            lines.append("  Law would predict. Transactions with these leading digits")
            lines.append("  may warrant closer review.")

    # Summary
    lines.append(section("SUMMARY"))
    sig_cols = [col for col, r in results.items() if r and r["significant"]]
    if sig_cols:
        lines.append("  Columns with statistically significant deviations:")
        for col in sig_cols:
            lines.append(f"    - {col}  (p = {results[col]['p_value']:.6f})")
        lines.append("")
        lines.append("  RECOMMENDATION: Review flagged transactions in benford_flagged.csv")
        lines.append("  These deviations do not prove fraud but indicate areas that may")
        lines.append("  benefit from further investigation.")
    else:
        lines.append("  No columns showed statistically significant deviations from")
        lines.append("  Benford's Law. The data appears consistent with expected")
        lines.append("  leading-digit distributions.")

    return "\n".join(lines) + "\n"


# ── Charts ───────────────────────────────────────────────────────────────────
def chart_benford(col: str, result: dict, out: Path):
    """Bar chart comparing observed vs expected Benford distribution."""
    if result is None:
        return

    digits = list(range(1, 10))
    observed = [result["observed_pct"][d] for d in digits]
    expected = [result["expected_pct"][d] for d in digits]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(digits))
    width = 0.35

    bars_obs = ax.bar(x - width / 2, observed, width, label="Observed", color=sns.color_palette()[0])
    bars_exp = ax.bar(x + width / 2, expected, width, label="Expected (Benford)", color=sns.color_palette()[2], alpha=0.7)

    ax.set_xlabel("Leading Digit")
    ax.set_ylabel("Frequency (%)")
    ax.set_title(f"Benford's Law Analysis: {col}")
    ax.set_xticks(x)
    ax.set_xticklabels(digits)
    ax.legend()

    # Add significance annotation
    sig_text = f"Chi² = {result['chi2_stat']:.2f}, p = {result['p_value']:.4f}"
    if result["significant"]:
        sig_text += "  ⚠ SIGNIFICANT DEVIATION"
    ax.annotate(sig_text, xy=(0.5, 0.97), xycoords="axes fraction",
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    # Highlight anomalous digits
    for d in result["flagged_digits"]:
        idx = d - 1
        ax.get_children()[idx].set_edgecolor("red")
        ax.get_children()[idx].set_linewidth(2)

    fig.tight_layout()
    fig.savefig(out / f"chart_benford_{col.lower()}.png", dpi=CHART_DPI)
    plt.close(fig)


def chart_benford_deviation(results: dict, out: Path):
    """Heatmap of per-digit deviations across all analyzed columns."""
    valid = {col: r for col, r in results.items() if r is not None}
    if not valid:
        return

    data = pd.DataFrame({col: r["deviation"] for col, r in valid.items()})
    data.index.name = "Leading Digit"

    fig, ax = plt.subplots(figsize=(max(6, len(valid) * 2), 5))
    sns.heatmap(data, annot=True, fmt=".1f", center=0, cmap="RdYlGn_r",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Deviation (pp)"})
    ax.set_title("Benford's Law Deviation Heatmap (Observed − Expected %)")
    ax.set_ylabel("Leading Digit")
    fig.tight_layout()
    fig.savefig(out / "chart_benford_heatmap.png", dpi=CHART_DPI)
    plt.close(fig)


# ── Flagged rows ─────────────────────────────────────────────────────────────
def extract_flagged_rows(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """
    Return rows where a numeric column's leading digit is flagged as
    anomalous, annotated with the reason.
    """
    flagged_frames = []

    for col, result in results.items():
        if result is None or not result["flagged_digits"]:
            continue
        digits_series = df[col].apply(leading_digit)
        mask = digits_series.isin(result["flagged_digits"])
        subset = df.loc[mask].copy()
        subset["_benford_column"] = col
        subset["_leading_digit"] = digits_series[mask]
        subset["_deviation_pp"] = subset["_leading_digit"].map(
            lambda d: result["deviation"].get(d, 0)
        )
        flagged_frames.append(subset)

    if not flagged_frames:
        return pd.DataFrame()
    return pd.concat(flagged_frames, ignore_index=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    if not os.path.exists(filepath):
        sys.exit(f"Error: file not found – {filepath}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Reading {filepath} …")
    df = pd.read_excel(filepath)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")

    # Determine which numeric columns to analyze
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Prioritize known accounting columns, then fall back to all numeric
    columns_to_analyze = [c for c in PRIORITY_COLUMNS if c in numeric_cols]
    remaining = [c for c in numeric_cols if c not in columns_to_analyze]
    columns_to_analyze.extend(remaining)

    if not columns_to_analyze:
        sys.exit("No numeric columns found for Benford analysis.")

    print(f"\nAnalyzing {len(columns_to_analyze)} numeric column(s): {columns_to_analyze}")

    # Run Benford analysis on each column
    results = {}
    for col in columns_to_analyze:
        print(f"  Analyzing: {col} …")
        results[col] = compute_benford(df[col])

    # ── Text report ──────────────────────────────────────────────────────
    report = build_report(results, filepath)
    report_path = OUTPUT_DIR / "benford_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")

    # ── JSON summary ─────────────────────────────────────────────────────
    summary = {}
    for col, r in results.items():
        if r is None:
            summary[col] = {"status": "skipped", "reason": "insufficient data"}
        else:
            summary[col] = {
                "n": r["n"],
                "chi2_stat": round(r["chi2_stat"], 4),
                "p_value": round(r["p_value"], 6),
                "significant": r["significant"],
                "flagged_digits": r["flagged_digits"],
            }
    summary_path = OUTPUT_DIR / "benford_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary saved to {summary_path}")

    # ── Charts ───────────────────────────────────────────────────────────
    for col, result in results.items():
        try:
            chart_benford(col, result, OUTPUT_DIR)
            if result:
                print(f"  ✓ chart_benford_{col.lower()}.png")
        except Exception as exc:
            print(f"  ✗ chart for {col}: {exc}")

    try:
        chart_benford_deviation(results, OUTPUT_DIR)
        print("  ✓ chart_benford_heatmap.png")
    except Exception as exc:
        print(f"  ✗ heatmap: {exc}")

    # ── Flagged rows ─────────────────────────────────────────────────────
    flagged = extract_flagged_rows(df, results)
    if not flagged.empty:
        flagged_path = OUTPUT_DIR / "benford_flagged.csv"
        flagged.to_csv(flagged_path, index=False)
        print(f"\nFlagged {len(flagged):,} rows → {flagged_path}")
    else:
        print("\nNo rows flagged for anomalous leading digits.")

    # ── Final status ─────────────────────────────────────────────────────
    sig_cols = [c for c, r in results.items() if r and r["significant"]]
    if sig_cols:
        print(f"\n⚠  {len(sig_cols)} column(s) show significant deviation from Benford's Law.")
        print("   Review benford_report.txt and benford_flagged.csv for details.")
    else:
        print("\n✓  All columns are consistent with Benford's Law.")

    print(f"\nDone. All outputs are in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
