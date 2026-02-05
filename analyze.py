#!/usr/bin/env python3
"""
Reusable Excel Data Analysis Script
====================================
Performs descriptive analysis on an Excel file and generates:
  - A text report  (output/analysis_report.txt)
  - Charts         (output/*.png)

Usage:
    python analyze.py                          # defaults to je_samples.xlsx
    python analyze.py path/to/other_file.xlsx  # analyse a different file
"""

import sys
import os
import textwrap
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_FILE = "je_samples.xlsx"
OUTPUT_DIR = Path("output")
CHART_DPI = 150
sns.set_theme(style="whitegrid", palette="muted")


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_number(n):
    """Format a number with commas."""
    if pd.isna(n):
        return "N/A"
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def section(title: str) -> str:
    """Return a section header for the text report."""
    line = "=" * 70
    return f"\n{line}\n  {title}\n{line}\n"


# ── Analysis functions ────────────────────────────────────────────────────────
def build_report(df: pd.DataFrame, filepath: str) -> str:
    """Generate the full text report and return it as a string."""
    lines: list[str] = []

    # ── 1. Overview ───────────────────────────────────────────────────────
    lines.append(section("1. FILE OVERVIEW"))
    lines.append(f"  Source file       : {filepath}")
    lines.append(f"  Analysis run at   : {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"  Total rows        : {fmt_number(len(df))}")
    lines.append(f"  Total columns     : {len(df.columns)}")

    # ── 2. Column inventory ──────────────────────────────────────────────
    lines.append(section("2. COLUMN INVENTORY"))
    lines.append(f"  {'Column':<30s} {'Dtype':<20s} {'Non-Null':>10s} {'Null':>8s} {'Null%':>7s}")
    lines.append("  " + "-" * 77)
    for col in df.columns:
        non_null = df[col].notna().sum()
        null = df[col].isna().sum()
        null_pct = null / len(df) * 100
        lines.append(
            f"  {col:<30s} {str(df[col].dtype):<20s} {non_null:>10,d} {null:>8,d} {null_pct:>6.1f}%"
        )

    # ── 3. Numeric summary ───────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        lines.append(section("3. NUMERIC COLUMN STATISTICS"))
        desc = df[numeric_cols].describe().T
        for col in desc.index:
            lines.append(f"\n  >> {col}")
            for stat in desc.columns:
                lines.append(f"     {stat:<8s}: {fmt_number(desc.loc[col, stat])}")

    # ── 4. Date columns ─────────────────────────────────────────────────
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    if date_cols:
        lines.append(section("4. DATE RANGES"))
        for col in date_cols:
            mn = df[col].min()
            mx = df[col].max()
            span = (mx - mn).days if pd.notna(mn) and pd.notna(mx) else "N/A"
            lines.append(f"  {col}")
            lines.append(f"     Min  : {mn}")
            lines.append(f"     Max  : {mx}")
            lines.append(f"     Span : {span} days")

    # ── 5. Categorical / text columns ────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if cat_cols:
        lines.append(section("5. CATEGORICAL / TEXT COLUMNS"))
        for col in cat_cols:
            nunique = df[col].nunique()
            lines.append(f"\n  >> {col}  (unique values: {fmt_number(nunique)})")
            if nunique <= 25:
                vc = df[col].value_counts().head(25)
                for val, cnt in vc.items():
                    lines.append(f"     {str(val):<40s} {cnt:>8,d}  ({cnt/len(df)*100:5.1f}%)")
            else:
                vc = df[col].value_counts().head(10)
                lines.append("     (Top 10 shown)")
                for val, cnt in vc.items():
                    lines.append(f"     {str(val):<40s} {cnt:>8,d}  ({cnt/len(df)*100:5.1f}%)")

    # ── 6. Duplicate & missing-data summary ──────────────────────────────
    lines.append(section("6. DATA QUALITY"))
    dup_rows = df.duplicated().sum()
    lines.append(f"  Fully duplicate rows : {fmt_number(dup_rows)} ({dup_rows/len(df)*100:.1f}%)")
    total_cells = len(df) * len(df.columns)
    total_null = df.isna().sum().sum()
    lines.append(f"  Total cells          : {fmt_number(total_cells)}")
    lines.append(f"  Total missing cells  : {fmt_number(total_null)} ({total_null/total_cells*100:.1f}%)")

    return "\n".join(lines) + "\n"


# ── Chart functions ───────────────────────────────────────────────────────────
def chart_missing_data(df: pd.DataFrame, out: Path):
    """Bar chart of missing-value percentages per column."""
    miss = df.isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0]
    if miss.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.35)))
    miss.plot.barh(ax=ax, color=sns.color_palette("Reds_r", len(miss)))
    ax.set_xlabel("% Missing")
    ax.set_title("Missing Data by Column")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "chart_missing_data.png", dpi=CHART_DPI)
    plt.close(fig)


def chart_numeric_distributions(df: pd.DataFrame, out: Path):
    """Histograms for key numeric columns."""
    # Pick numeric columns that have reasonable variance
    num_cols = [c for c in df.select_dtypes("number").columns
                if df[c].notna().sum() > 100 and df[c].std() > 0]
    # Limit to most interesting columns
    priority = ["Amount", "AbsoluteAmount", "Debit", "Credit"]
    cols = [c for c in priority if c in num_cols]
    if not cols:
        cols = num_cols[:4]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(data, bins=50, edgecolor="white", linewidth=0.4)
        ax.set_title(col)
        ax.set_ylabel("Frequency")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Numeric Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "chart_numeric_distributions.png", dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)


def chart_top_categories(df: pd.DataFrame, out: Path):
    """Bar charts for key categorical columns."""
    cat_targets = ["Source", "BusinessUnit", "AccountType", "AccountClass", "PreparerID"]
    cols = [c for c in cat_targets if c in df.columns and df[c].notna().sum() > 0]
    if not cols:
        return
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        vc = df[col].value_counts().head(10)
        vc.plot.barh(ax=ax)
        ax.set_title(f"Top {col}")
        ax.invert_yaxis()
    fig.suptitle("Categorical Breakdowns", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "chart_top_categories.png", dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)


def chart_timeline(df: pd.DataFrame, out: Path):
    """Line chart of transaction volume over time."""
    date_col = None
    for candidate in ["EffectiveDate", "EntryDate"]:
        if candidate in df.columns and pd.api.types.is_datetime64_any_dtype(df[candidate]):
            date_col = candidate
            break
    if date_col is None:
        return
    monthly = df.groupby(df[date_col].dt.to_period("M")).size()
    monthly.index = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=1.5)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15)
    ax.set_title(f"Transaction Volume by Month ({date_col})")
    ax.set_ylabel("Row Count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "chart_timeline.png", dpi=CHART_DPI)
    plt.close(fig)


def chart_debit_credit_balance(df: pd.DataFrame, out: Path):
    """Compare total debits vs credits."""
    debit = pd.to_numeric(df.get("Debit"), errors="coerce").sum()
    credit = pd.to_numeric(df.get("Credit"), errors="coerce").sum()
    if pd.isna(debit) and pd.isna(credit):
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Total Debits", "Total Credits"], [debit or 0, credit or 0],
                  color=[sns.color_palette()[0], sns.color_palette()[2]])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Debit vs Credit Totals")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    fig.savefig(out / "chart_debit_credit.png", dpi=CHART_DPI)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    if not os.path.exists(filepath):
        sys.exit(f"Error: file not found – {filepath}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Reading {filepath} …")
    df = pd.read_excel(filepath)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")

    # ── Text report ──────────────────────────────────────────────────────
    report = build_report(df, filepath)
    report_path = OUTPUT_DIR / "analysis_report.txt"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")

    # ── Charts ───────────────────────────────────────────────────────────
    chart_funcs = [
        chart_missing_data,
        chart_numeric_distributions,
        chart_top_categories,
        chart_timeline,
        chart_debit_credit_balance,
    ]
    for fn in chart_funcs:
        try:
            fn(df, OUTPUT_DIR)
            print(f"  ✓ {fn.__name__}")
        except Exception as exc:
            print(f"  ✗ {fn.__name__}: {exc}")

    print(f"\nDone. All outputs are in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
