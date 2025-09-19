#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce regime_plot.R in Python, generating scenario_plot.pdf and scenario_plot.png.

Dependencies:
    pip install pandas numpy matplotlib xlrd==1.2.0 openpyxl

Inputs:
    ie_data.xls  (the same Shiller workbook used by the R script; place beside this file)

Notes:
- Reimplements the R helpers used in your code path:
    - shiller_to_xts()  -> shiller_to_dataframe()
    - ifna(), ind.roc() -> fillna(0), pct_change()
    - create_weight_matrix(..., .6/.4 monthly) -> constant monthly weights
    - xts.mat("rowSums2", ...) -> row-wise dot product
    - RollingMean(..., window=3, na_method='ignore') -> rolling(3, min_periods=1).mean()
    - ram.colors(10) -> fixed hex palette below (chosen to match the figure)
    - ReSolve.Axis.Theme/ReSolve.Axis.Adjust -> explicit Matplotlib styling

Outputs:
    scenario_plot.pdf
    scenario_plot.png
"""

import math
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FuncFormatter

# ---------------------------
# 1) Data ingestion utilities
# ---------------------------

def shiller_to_dataframe(file_path: str, date_col=0, return_col=9, bond_return_col=18):
    """
    Read Shiller's xls (same file as R), sheet index equivalent to R's 'sheet=5'.

    R's readxl is 1-indexed (sheet=5). Pandas is 0-indexed.
    We first try sheet_name=4; if that fails, we fall back to 5.
    The R code keeps rows from 7:n (1-based), i.e., drop first 6 rows.

    The R converts year-month as 'YYYY.MM' where '.1' means '.10' (October).
    We replicate that parsing logic exactly, then build a proper Date column.

    Returns
    -------
    pandas.DataFrame with columns:
        Date (datetime64[ns])
        total_return_price (float)
        total_bond_returns (float)
    """
    # Try to read the sheet the same way (R sheet=5 -> Python sheet index=4)
    def _read_sheet_df(sheet_ref):
        """Read a sheet (by index or name) to a DataFrame, fallback to xlrd for legacy .xls."""
        try:
            # Try pandas first
            return pd.read_excel(file_path, sheet_name=sheet_ref, header=None)
        except Exception as e:
            msg = str(e)
            # Fallback path: use xlrd to read legacy .xls when pandas enforces xlrd>=2
            try:
                import xlrd  # type: ignore
                wb = xlrd.open_workbook(file_path)
                if isinstance(sheet_ref, int):
                    sh = wb.sheet_by_index(sheet_ref)
                else:
                    sh = wb.sheet_by_name(sheet_ref)
                rows = []
                for r in range(sh.nrows):
                    row_vals = [sh.cell_value(r, c) for c in range(sh.ncols)]
                    rows.append(row_vals)
                return pd.DataFrame(rows)
            except Exception as e2:
                # Chain the original pandas error context
                raise RuntimeError(f"pandas read_excel failed: {msg}; xlrd fallback failed: {e2}")

    tried = []
    df = None
    last_err = None
    # Prefer a sheet named 'Data' if present, else try legacy indices 4/5, then 1.
    for sheet_ref in ("Data", 4, 5, 1):
        try:
            tried.append(sheet_ref)
            df = _read_sheet_df(sheet_ref)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"Failed to read '{file_path}' from sheet indices {tried}: {last_err}")

    # Drop the first 6 rows (R took [7:nrow(data), ])
    df = df.iloc[6:].copy()

    # Ensure we have enough columns
    max_needed = max(date_col, return_col, bond_return_col)
    if df.shape[1] <= max_needed:
        raise ValueError(
            f"Sheet does not have the required columns. "
            f"Need at least {max_needed+1} columns, found {df.shape[1]}."
        )

    # Extract the three columns we need
    # Convert the date column to strings emulating the R rounding + textual fix
    #  - round numeric to 2 decimals
    #  - replace trailing ".1" with ".10"
    raw_dates = df.iloc[:, date_col].astype(str).tolist()
    # In R: round(as.numeric(...), 2) first.
    # We'll try to coerce to float, round(2), then convert back to string.
    def clean_date_cell(x):
        try:
            f = float(x)
            s = f"{round(f, 2)}"
        except Exception:
            s = str(x)
        # Apply the exact gsub("\\.1$", ".10", ...)
        if s.endswith(".1"):
            s = s[:-2] + ".10"
        return s

    cleaned = [clean_date_cell(x) for x in raw_dates]
    # Now append "-01" (first day of month) and parse with format "%Y.%m-%d"
    # Example: "1871.10-01" -> 1871-10-01
    def to_date(s):
        try:
            return datetime.strptime(s + "-01", "%Y.%m-%d")
        except Exception:
            return pd.NaT

    dates = [to_date(s) for s in cleaned]

    total_return_price = pd.to_numeric(df.iloc[:, return_col], errors="coerce")
    total_bond_returns = pd.to_numeric(df.iloc[:, bond_return_col], errors="coerce")

    # If the bond column looks like a monthly gross return factor (~1.x),
    # convert it to a level index via cumulative product so pct_change works below.
    vals = total_bond_returns.dropna()
    if len(vals) > 5:
        median_v = float(vals.median())
        max_v = float(vals.max())
        if 0.8 < median_v < 1.2 and max_v < 2.5:
            # treat as gross return factor
            total_bond_returns = (total_bond_returns.fillna(1.0)).cumprod()

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "total_return_price": total_return_price,
            "total_bond_returns": total_bond_returns,
        }
    ).dropna(subset=["Date"]).reset_index(drop=True)

    return out


# -----------------------------------
# 2) Portfolio construction utilities
# -----------------------------------

def compute_returns_from_levels(df_levels: pd.DataFrame, cols):
    """
    Equivalent to ind.roc (simple ratio returns): r_t = level_t / level_{t-1} - 1
    Fill initial NaNs with 0 (ifna(..., 0)).
    """
    ret = df_levels[cols].pct_change()
    ret = ret.fillna(0.0)
    return ret

def rolling_mean(series: pd.Series, window=3):
    """
    Rolling mean with min_periods=1 (na_method='ignore'),
    then replace the first two values by the original series values
    to match the R code behavior.
    """
    r = series.rolling(window=window, min_periods=1).mean()
    if len(series) >= 2:
        r.iloc[0] = series.iloc[0]
        r.iloc[1] = series.iloc[1]
    return r

def ram_colors(n=10):
    """
    ReSolve-inspired palette (fixed hex values chosen to visually match the provided figure):
      1: muted steel/denim blue (Inflationary Growth)
      2: warm sand yellow (Stagflation)
      3: neutral dark grey (Disinflationary Growth)
      4: light blue/cyan (Deflationary Bust)
      5: lavender (Stagflation ???)
    Remaining slots filled with reasonable complements to keep n=10 stable.
    """
    palette = [
        "#6C87B8",  # 1
        "#E8C982",  # 2
        "#8E8E8E",  # 3
        "#BBD7F4",  # 4
        "#B9A5D8",  # 5
        "#5E9FA7",  # 6 (unused)
        "#C2C2C2",  # 7 (unused)
        "#A6B9D0",  # 8 (unused)
        "#D5E6F9",  # 9 (unused)
        "#7A92B8",  # 10 (unused)
    ]
    if n <= len(palette):
        return palette[:n]
    else:
        # repeat if more requested
        k = (n + len(palette) - 1) // len(palette)
        return (palette * k)[:n]

# -----------------------------------
# 3) Replicate the R pipeline
# -----------------------------------

def main():
    # ---- Read the data exactly like the R code ----
    xfile = "ie_data.xls"
    if not os.path.exists(xfile):
        raise FileNotFoundError(
            f"'{xfile}' not found. Place the same Shiller workbook (ie_data.xls) next to this script."
        )

    # date_col = 1 in R -> 0-based here; return_col=10 -> 9; bond_return_col=19 -> 18
    df = shiller_to_dataframe(
        xfile, date_col=0, return_col=9, bond_return_col=18
    )

    # Keep as monthly time series indexed by Date
    df = df.sort_values("Date").set_index("Date")

    # ---- Compute returns like R: ifna(ind.roc(xts_data), 0) ----
    # xts_data had [total_return_price, total_bond_returns]
    returns = compute_returns_from_levels(df, ["total_return_price", "total_bond_returns"])

    # R: returns = returns["1900-02::"]
    returns = returns.loc[returns.index >= pd.Timestamp("1900-02-01")]

    # Create a constant 60/40 weight each month (rebalance monthly)
    w = np.array([0.60, 0.40])
    w = np.tile(w, (len(returns), 1))  # constant per-row weights

    # Row-wise dot product -> portfolio return
    port_ret = (returns.values * w).sum(axis=1)
    port_ret = pd.Series(port_ret, index=returns.index, name="Balanced")

    # R:
    # real.equity = 100 * cumprod(1 + rowSums2(weights*returns))
    real_equity = 100.0 * (1.0 + port_ret).cumprod()

    # Prepend top_row=100 at 1900-01-01
    start_anchor = pd.Series(
        [100.0], index=[pd.Timestamp("1900-01-01")], name="Balanced"
    )
    real_equity = pd.concat([start_anchor, real_equity])
    real_equity = real_equity.sort_index()

    # Rolling 3-month mean with first two values restored to original
    real_equity_roll = rolling_mean(real_equity, window=3)
    real_equity = real_equity_roll.rename("Balanced")

    # ------------------------------
    # 4) Scenario frames + aesthetics
    # ------------------------------
    # Terminal date = last index of real_equity
    terminal_date = real_equity.index[-1].date()

    # Same scenario segments as R
    dates = [
        (pd.Timestamp("1900-01-01"), pd.Timestamp("1917-01-01")),  # 1 Inflationary Growth
        (pd.Timestamp("1917-01-01"), pd.Timestamp("1921-01-01")),  # 2 Deflationary Bust
        (pd.Timestamp("1921-01-01"), pd.Timestamp("1929-08-01")),  # 3 Dis-Inflationary Growth
        (pd.Timestamp("1929-08-01"), pd.Timestamp("1949-06-01")),  # 4 Deflationary Bust
        (pd.Timestamp("1949-06-01"), pd.Timestamp("1964-01-01")),  # 5 Disinflationary Growth
        (pd.Timestamp("1964-01-01"), pd.Timestamp("1969-01-01")),  # 6 Inflationary Growth
        (pd.Timestamp("1969-01-01"), pd.Timestamp("1982-06-01")),  # 7 Stagflation
        (pd.Timestamp("1982-06-01"), pd.Timestamp("2000-01-01")),  # 8 Disinflationary Growth
        (pd.Timestamp("2000-01-01"), pd.Timestamp("2003-03-01")),  # 9 Deflationary Bust
        (pd.Timestamp("2003-03-01"), pd.Timestamp("2007-10-01")),  # 10 Inflationary Growth
        (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-03-01")),  # 11 Deflationary Bust
        (pd.Timestamp("2009-03-01"), pd.Timestamp("2021-12-01")),  # 12 Disinflationary Growth
        (pd.Timestamp("2021-12-01"), pd.Timestamp(terminal_date)), # 13 Stagflation ???
    ]

    scenario = [
        "Inflationary\nGrowth",
        "Deflationary Bust",
        "Dis-\nInflationary\nGrowth",
        "Deflationary Bust",
        "Disinflationary\nGrowth",
        "Inflationary Growth",
        "Stagflation",
        "  Disinflationary\nGrowth",
        "Deflationary Bust",
        "Inflationary Growth",
        "Deflationary Bust",
        "Disinflationary\nGrowth",
        "Stagflation ???",
    ]

    # Orientation exactly as in R (H vs V)
    orientation = ["H"] * 13
    for i in [1, 5, 8, 9, 10, 12]:  # 0-based indices for {2,6,9,10,11,13}
        orientation[i] = "V"

    # ReSolve colors mapped exactly as in the R vector using ram.colors(10)[...]
    pal = ram_colors(10)
    regime_colors = [
        pal[0],  # 1
        pal[3],  # 4
        pal[2],  # 3
        pal[3],  # 4
        pal[2],  # 3
        pal[0],  # 1
        pal[1],  # 2
        pal[2],  # 3
        pal[3],  # 4
        pal[0],  # 1
        pal[3],  # 4
        pal[2],  # 3
        pal[4],  # 5
    ]

    scen_df = pd.DataFrame(dates, columns=["Start", "End"])
    scen_df["Scenario"] = scenario
    scen_df["orientation"] = orientation
    scen_df["Mid"] = scen_df["Start"] + (scen_df["End"] - scen_df["Start"]) / 2.0
    scen_df["Color"] = regime_colors

    # --------------------------------
    # 5) Log2 scale and tick formatting
    # --------------------------------
    # new_max_y = max(Balanced) * 3.2 (match R)
    new_max_y = real_equity.max() * 3.2
    min_y = real_equity.min()

    def log2_breaks(vmin, vmax):
        lo = int(math.floor(math.log2(max(vmin, 1e-12))))
        hi = int(math.ceil(math.log2(vmax)))
        return [2 ** k for k in range(lo, hi + 1)]

    yticks = log2_breaks(min_y, new_max_y)

    def dollar_fmt(x, pos):
        try:
            return f"${int(round(x)):,.0f}"
        except Exception:
            return ""

    # -----------------------
    # 6) Plot (Matplotlib API)
    # -----------------------
    mpl.rcParams["figure.dpi"] = 150
    # Prefer Helvetica Neue if available; set sans-serif stack explicitly
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Sans-Serif",
    ]

    fig = plt.figure(figsize=(12, 5), constrained_layout=False)
    ax = plt.gca()

    # Minimal theme base: remove grid, top/right spines; heavy left/bottom spines
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Draw regime rectangles first (so the line is on top), using alpha=0.8 for brighter colors
    for _, r in scen_df.iterrows():
        ax.axvspan(r["Start"], r["End"], ymin=0, ymax=1, color=r["Color"], alpha=0.8, lw=0)

    # Plot the cumulative equity line
    ax.plot(real_equity.index, real_equity.values, color="black", linewidth=2.0)

    # Add scenario labels - ALL aligned to top of chart
    label_y = new_max_y * 0.95  # Same position for ALL labels

    for _, r in scen_df.iterrows():
        if r["orientation"] == "H":
            ax.text(
                r["Mid"],
                label_y,
                r["Scenario"],
                rotation=0,
                ha="center",
                va="top",  # Align top edge to y-coordinate
                fontsize=9,
            )
        else:  # vertical - center horizontally in regime
            ax.text(
                r["Mid"],
                label_y,
                r["Scenario"],
                rotation=90,
                rotation_mode='anchor',  # Rotate around anchor point
                ha="right",  # Becomes bottom-aligned after rotation
                va="center",  # Center horizontally in the regime
                fontsize=9,
            )

    # X axis: ticks every 10 years, labels as YYYY, with tight margins like ggplot expand=c(0.005,0)
    ax.set_xlim(real_equity.index.min(), real_equity.index.max())
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Reduce default margins to mimic ggplot's expand=c(0.005, 0)
    ax.margins(x=0.0, y=0.02)

    # Y axis: log2 scale with exact breaks and dollar labels; explicit bounds
    # Matplotlib log scale with base 2:
    ax.set_yscale("log", base=2)
    ax.set_ylim([min_y, new_max_y])

    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))

    # Heavy axis lines and ticks per your custom ReSolve theme
    # (thick left/bottom spines; longer/wider ticks; black tick params)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(3.0)
        ax.spines[side].set_color("black")

    # Add white tick marks on the y-axis spine to create divisions
    # These need to extend across the full width of the thick spine
    x_min = ax.get_xlim()[0]
    spine_width_data = 0.0025 * (ax.get_xlim()[1] - ax.get_xlim()[0])  # Slightly reduced width
    for y_val in yticks:
        if min_y <= y_val <= new_max_y:
            ax.plot([x_min - spine_width_data, x_min + spine_width_data * 0.8], [y_val, y_val],
                   color='white', linewidth=2, clip_on=False, zorder=10)

    ax.tick_params(
        axis="both",
        which="major",
        length=0,  # Remove tick marks
        width=0,
        color="black",
        labelsize=11,
        direction="out",
        pad=4,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=0,  # Remove tick marks
        width=0,
        color="black",
        direction="out",
    )

    # Remove extra padding so axes hug the figure like your example
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.08)

    # Save to PDF and PNG to match your ggsave calls
    fig.savefig("scenario_plot.pdf", dpi=300)
    fig.savefig("scenario_plot.png", dpi=300)

    # Export CSV with raw and final values on common date index
    # Create a DataFrame with the final equity values
    export_df = pd.DataFrame(
        {"60_40_cumulative_equity": real_equity.values},
        index=real_equity.index
    )

    # Add raw values from original data (matching dates)
    raw_data = df.copy()
    raw_data.index = pd.to_datetime(raw_data.index)
    export_df = export_df.join(raw_data[["total_return_price", "total_bond_returns"]], how="left")

    # Add the portfolio returns (align by index)
    port_ret_df = port_ret.to_frame("60_40_portfolio_return")
    export_df = export_df.join(port_ret_df, how="left")

    # Reorder and rename columns for clarity
    export_df = export_df[
        ["total_return_price", "total_bond_returns", "60_40_portfolio_return", "60_40_cumulative_equity"]
    ]
    export_df.columns = [
        "raw_stock_price_level",
        "raw_bond_price_level",
        "60_40_portfolio_return",
        "60_40_cumulative_equity"
    ]

    # Save to CSV
    export_df.to_csv("scenario_plot_data.csv")
    print("Saved: scenario_plot.pdf, scenario_plot.png, scenario_plot_data.csv")

if __name__ == "__main__":
    main()
