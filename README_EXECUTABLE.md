# Shiller Regime Plot Generator - Executable Version

## For End Users (Your Friend)

### Windows Users
1. Download the `ShillerRegimePlot.exe` file
2. Double-click to run it
3. The program will:
   - Download the latest Shiller data from shillerdata.com
   - Generate three output files in the same folder:
     - `scenario_plot.png` - The regime plot image
     - `scenario_plot.pdf` - The regime plot in PDF format
     - `scenario_plot_data.csv` - The underlying data

### Mac/Linux Users
1. Download the `ShillerRegimePlot` file
2. Open Terminal and navigate to the download location
3. Make it executable: `chmod +x ShillerRegimePlot`
4. Run it: `./ShillerRegimePlot`

## For Developers (Building the Executable)

### Prerequisites
- Python 3.8 or later
- pip (Python package manager)

### Building Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements_exe.txt
   ```

2. **Test the standalone script:**
   ```bash
   python shiller_regime_plot_standalone.py
   ```

3. **Build the executable:**
   ```bash
   python build_exe.py
   ```

4. **Find the executable:**
   - Windows: `dist/ShillerRegimePlot.exe`
   - Mac/Linux: `dist/ShillerRegimePlot`

### What the Executable Does

1. **Downloads Latest Data:**
   - Fetches the current download URL from shillerdata.com
   - Downloads the latest ie_data.xls file (updated through current month)
   - This data is more recent than Yale's official site (which only has through 2023)

2. **Processes Data:**
   - Reads the Excel file using xlrd
   - Calculates 60/40 portfolio returns (60% stocks, 40% bonds)
   - Identifies market regime periods
   - Creates cumulative equity curve

3. **Generates Output:**
   - Creates a multi-colored regime plot showing different market periods
   - Exports the plot as both PNG and PDF
   - Saves the underlying data to CSV

### Troubleshooting

**"File not found" errors:**
- Ensure the executable has internet access to download data
- Check firewall settings

**"Cannot create file" errors:**
- Ensure the executable has write permissions in its directory
- Try running as administrator (Windows) or with sudo (Mac/Linux)

**Plot looks wrong:**
- The data source may have changed format
- Check that shillerdata.com is accessible

### File Sizes
- Standalone script: ~15 KB
- Executable: ~50-100 MB (includes Python runtime and all dependencies)
- Downloaded data: ~1.6 MB
- Output files: ~200 KB total

### Security Notes
- The executable is not code-signed, so you may get security warnings
- Windows: Click "More info" then "Run anyway" in SmartScreen
- Mac: Right-click and select "Open" to bypass Gatekeeper
- The program only downloads from shillerdata.com and creates local files