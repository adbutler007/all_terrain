#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Shiller Regime Plot Generator
Downloads latest data from shillerdata.com and generates regime plot

This single file can be converted to an executable for distribution.
"""

import os
import sys
import re
import math
from datetime import datetime, timedelta

# Set matplotlib backend before importing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better compatibility

# Check and install required packages
def check_and_install_packages():
    """Check for required packages and install if missing."""
    required = {
        'requests': 'requests',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'xlrd': 'xlrd==1.2.0'  # Specific version for .xls support
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("Missing required packages. Installing...")
        import subprocess
        for package in missing:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("Installation complete.\n")

# Run package check
check_and_install_packages()

# Now import the packages
import requests
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FuncFormatter

# ===========================
# Data Download Functions
# ===========================

def get_latest_download_url(verbose=True):
    """Fetch the latest download URL from shillerdata.com."""
    try:
        if verbose:
            print("Fetching latest download URL from shillerdata.com...")

        response = requests.get("https://shillerdata.com", timeout=10)
        response.raise_for_status()

        # Look for the download link pattern
        pattern = r'//img1\.wsimg\.com/blobby/go/[^"\']+?ie_data\.xls[^"\'>]*'
        match = re.search(pattern, response.text)

        if match:
            url = match.group(0)
            if url.startswith('//'):
                url = 'https:' + url
            if verbose:
                print(f"Found download URL")
            return url
        else:
            if verbose:
                print("Could not find download link on shillerdata.com")
            return None

    except Exception as e:
        if verbose:
            print(f"Error fetching download URL: {e}")
        return None


def download_ie_data(output_file="ie_data.xls", verbose=True):
    """Download the updated ie_data.xls file."""
    url = get_latest_download_url(verbose)

    if not url:
        print("Failed to get download URL from shillerdata.com")
        return False

    try:
        if verbose:
            print(f"Downloading updated Shiller data...")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        with open(output_file, 'wb') as f:
            f.write(response.content)

        file_size = os.path.getsize(output_file)
        if verbose:
            print(f"Successfully downloaded {output_file}")
            print(f"File size: {file_size:,} bytes\n")

        if file_size < 1000000:
            print("Warning: File seems unusually small.")
            return False

        return True

    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


# ===========================
# Data Processing Functions
# ===========================

def shiller_to_dataframe(file_path: str, date_col=0, return_col=9, bond_return_col=18):
    """Read Shiller's xls file and convert to DataFrame."""
    def _read_sheet_df(sheet_ref):
        try:
            return pd.read_excel(file_path, sheet_name=sheet_ref, header=None)
        except:
            try:
                import xlrd
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
            except Exception as e:
                raise RuntimeError(f"Failed to read Excel file: {e}")

    df = None
    for sheet_ref in ("Data", 4, 5, 1):
        try:
            df = _read_sheet_df(sheet_ref)
            break
        except:
            df = None

    if df is None:
        raise RuntimeError(f"Failed to read '{file_path}'")

    df = df.iloc[6:].copy()

    max_needed = max(date_col, return_col, bond_return_col)
    if df.shape[1] <= max_needed:
        raise ValueError(f"Sheet does not have required columns")

    raw_dates = df.iloc[:, date_col].astype(str).tolist()

    def clean_date_cell(x):
        try:
            f = float(x)
            s = f"{round(f, 2)}"
        except:
            s = str(x)
        if s.endswith(".1"):
            s = s[:-2] + ".10"
        return s

    cleaned = [clean_date_cell(x) for x in raw_dates]

    def to_date(s):
        try:
            return datetime.strptime(s + "-01", "%Y.%m-%d")
        except:
            return pd.NaT

    dates = [to_date(s) for s in cleaned]

    total_return_price = pd.to_numeric(df.iloc[:, return_col], errors="coerce")
    total_bond_returns = pd.to_numeric(df.iloc[:, bond_return_col], errors="coerce")

    vals = total_bond_returns.dropna()
    if len(vals) > 5:
        median_v = float(vals.median())
        max_v = float(vals.max())
        if 0.8 < median_v < 1.2 and max_v < 2.5:
            total_bond_returns = (total_bond_returns.fillna(1.0)).cumprod()

    out = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "total_return_price": total_return_price,
        "total_bond_returns": total_bond_returns,
    }).dropna(subset=["Date"]).reset_index(drop=True)

    return out


def compute_returns_from_levels(df_levels: pd.DataFrame, cols):
    """Compute returns from price levels."""
    ret = df_levels[cols].pct_change()
    ret = ret.fillna(0.0)
    return ret


def rolling_mean(series: pd.Series, window=3):
    """Rolling mean with special handling for first values."""
    r = series.rolling(window=window, min_periods=1).mean()
    if len(series) >= 2:
        r.iloc[0] = series.iloc[0]
        r.iloc[1] = series.iloc[1]
    return r


def ram_colors(n=10):
    """ReSolve-inspired color palette."""
    palette = [
        "#6C87B8",  # 1 - Inflationary Growth
        "#E8C982",  # 2 - Stagflation
        "#8E8E8E",  # 3 - Disinflationary Growth
        "#BBD7F4",  # 4 - Deflationary Bust
        "#B9A5D8",  # 5 - Stagflation ???
        "#5E9FA7",  # 6
        "#C2C2C2",  # 7
        "#A6B9D0",  # 8
        "#D5E6F9",  # 9
        "#7A92B8",  # 10
    ]
    if n <= len(palette):
        return palette[:n]
    else:
        k = (n + len(palette) - 1) // len(palette)
        return (palette * k)[:n]


# ===========================
# Main Processing Function
# ===========================

def create_regime_plot():
    """Main function to create the regime plot."""
    print("Processing data and creating plots...")

    # Read the data
    xfile = "ie_data.xls"
    if not os.path.exists(xfile):
        raise FileNotFoundError(f"'{xfile}' not found. Download failed?")

    df = shiller_to_dataframe(xfile, date_col=0, return_col=9, bond_return_col=18)
    df = df.sort_values("Date").set_index("Date")

    # Compute returns
    returns = compute_returns_from_levels(df, ["total_return_price", "total_bond_returns"])
    returns = returns.loc[returns.index >= pd.Timestamp("1900-02-01")]

    # Create 60/40 portfolio
    w = np.array([0.60, 0.40])
    w = np.tile(w, (len(returns), 1))
    port_ret = (returns.values * w).sum(axis=1)
    port_ret = pd.Series(port_ret, index=returns.index, name="Balanced")

    # Calculate cumulative equity
    real_equity = 100.0 * (1.0 + port_ret).cumprod()
    start_anchor = pd.Series([100.0], index=[pd.Timestamp("1900-01-01")], name="Balanced")
    real_equity = pd.concat([start_anchor, real_equity])
    real_equity = real_equity.sort_index()
    real_equity_roll = rolling_mean(real_equity, window=3)
    real_equity = real_equity_roll.rename("Balanced")

    # Define scenario frames
    terminal_date = real_equity.index[-1].date()

    dates = [
        (pd.Timestamp("1900-01-01"), pd.Timestamp("1917-01-01")),
        (pd.Timestamp("1917-01-01"), pd.Timestamp("1921-01-01")),
        (pd.Timestamp("1921-01-01"), pd.Timestamp("1929-08-01")),
        (pd.Timestamp("1929-08-01"), pd.Timestamp("1949-06-01")),
        (pd.Timestamp("1949-06-01"), pd.Timestamp("1964-01-01")),
        (pd.Timestamp("1964-01-01"), pd.Timestamp("1969-01-01")),
        (pd.Timestamp("1969-01-01"), pd.Timestamp("1982-06-01")),
        (pd.Timestamp("1982-06-01"), pd.Timestamp("2000-01-01")),
        (pd.Timestamp("2000-01-01"), pd.Timestamp("2003-03-01")),
        (pd.Timestamp("2003-03-01"), pd.Timestamp("2007-10-01")),
        (pd.Timestamp("2007-10-01"), pd.Timestamp("2009-03-01")),
        (pd.Timestamp("2009-03-01"), pd.Timestamp("2021-12-01")),
        (pd.Timestamp("2021-12-01"), pd.Timestamp(terminal_date)),
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

    orientation = ["H"] * 13
    for i in [1, 5, 8, 9, 10, 12]:
        orientation[i] = "V"

    pal = ram_colors(10)
    regime_colors = [
        pal[0], pal[3], pal[2], pal[3], pal[2], pal[0],
        pal[1], pal[2], pal[3], pal[0], pal[3], pal[2], pal[4],
    ]

    scen_df = pd.DataFrame(dates, columns=["Start", "End"])
    scen_df["Scenario"] = scenario
    scen_df["orientation"] = orientation
    scen_df["Mid"] = scen_df["Start"] + (scen_df["End"] - scen_df["Start"]) / 2.0
    scen_df["Color"] = regime_colors

    # Create the plot
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
        except:
            return ""

    # Set up matplotlib
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Helvetica Neue", "Helvetica", "Arial",
        "DejaVu Sans", "Liberation Sans", "Sans-Serif",
    ]

    fig = plt.figure(figsize=(12, 5), constrained_layout=False)
    ax = plt.gca()

    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Draw regime rectangles
    for _, r in scen_df.iterrows():
        ax.axvspan(r["Start"], r["End"], ymin=0, ymax=1, color=r["Color"], alpha=0.8, lw=0)

    # Plot the line
    ax.plot(real_equity.index, real_equity.values, color="black", linewidth=2.0)

    # Add labels
    label_y = new_max_y * 0.95
    for _, r in scen_df.iterrows():
        if r["orientation"] == "H":
            ax.text(r["Mid"], label_y, r["Scenario"], rotation=0,
                   ha="center", va="top", fontsize=9)
        else:
            ax.text(r["Mid"], label_y, r["Scenario"], rotation=90,
                   rotation_mode='anchor', ha="right", va="center", fontsize=9)

    # Configure axes
    ax.set_xlim(real_equity.index.min(), real_equity.index.max())
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.0, y=0.02)

    ax.set_yscale("log", base=2)
    ax.set_ylim([min_y, new_max_y])
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))

    # Style the axes
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(3.0)
        ax.spines[side].set_color("black")

    # Add white tick marks on y-axis
    x_min = ax.get_xlim()[0]
    spine_width_data = 0.0025 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    for y_val in yticks:
        if min_y <= y_val <= new_max_y:
            ax.plot([x_min - spine_width_data, x_min + spine_width_data * 0.8], [y_val, y_val],
                   color='white', linewidth=2, clip_on=False, zorder=10)

    ax.tick_params(axis="both", which="major", length=0, width=0,
                  color="black", labelsize=11, direction="out", pad=4)
    ax.tick_params(axis="both", which="minor", length=0, width=0,
                  color="black", direction="out")

    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.08)

    # Save outputs
    fig.savefig("scenario_plot.pdf", dpi=300)
    fig.savefig("scenario_plot.png", dpi=300)

    # Export CSV
    export_df = pd.DataFrame(
        {"60_40_cumulative_equity": real_equity.values},
        index=real_equity.index
    )
    raw_data = df.copy()
    raw_data.index = pd.to_datetime(raw_data.index)
    export_df = export_df.join(raw_data[["total_return_price", "total_bond_returns"]], how="left")
    port_ret_df = port_ret.to_frame("60_40_portfolio_return")
    export_df = export_df.join(port_ret_df, how="left")
    export_df = export_df[
        ["total_return_price", "total_bond_returns", "60_40_portfolio_return", "60_40_cumulative_equity"]
    ]
    export_df.columns = [
        "raw_stock_price_level", "raw_bond_price_level",
        "60_40_portfolio_return", "60_40_cumulative_equity"
    ]
    export_df.to_csv("scenario_plot_data.csv")

    print("Successfully created:")
    print("  - scenario_plot.pdf")
    print("  - scenario_plot.png")
    print("  - scenario_plot_data.csv")


# ===========================
# Main Entry Point
# ===========================

def main():
    """Main function to coordinate download and plot generation."""
    print("=" * 60)
    print("Shiller Regime Plot Generator")
    print("=" * 60)
    print()

    # Download the data
    if not download_ie_data():
        print("\nFailed to download data. Exiting.")
        sys.exit(1)

    # Create the plots
    try:
        create_regime_plot()
        print("\nProcess completed successfully!")
    except Exception as e:
        print(f"\nError creating plots: {e}")
        sys.exit(1)

    # Wait for user input before closing (useful for .exe)
    print("\nPress Enter to exit...")
    input()


if __name__ == "__main__":
    main()