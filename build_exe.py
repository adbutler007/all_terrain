#!/usr/bin/env python3
"""
Build script to create standalone executable for Shiller Regime Plot Generator

Usage:
    python build_exe.py
"""

import os
import sys
import subprocess
import shutil


def main():
    """Build the executable using PyInstaller."""
    print("=" * 60)
    print("Building Shiller Regime Plot Executable")
    print("=" * 60)
    print()

    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        import PyInstaller

    # Clean up previous builds
    for dir_name in ["build", "dist"]:
        if os.path.exists(dir_name):
            print(f"Removing old {dir_name} directory...")
            shutil.rmtree(dir_name)

    # Build command for PyInstaller
    script_name = "shiller_regime_plot_standalone.py"
    exe_name = "ShillerRegimePlot"

    # PyInstaller options:
    # --onefile: Create a single executable file
    # --windowed: Don't show console window (remove for debugging)
    # --name: Name of the executable
    # --clean: Clean PyInstaller cache
    # --noconfirm: Replace output directory without asking

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",           # Single file executable
        "--console",           # Show console (so they can see progress)
        f"--name={exe_name}",  # Executable name
        "--clean",             # Clean build
        "--noconfirm",         # Don't ask for confirmation
        "--hidden-import=matplotlib.backends.backend_pdf",  # Include PDF backend
        "--hidden-import=matplotlib.backends.backend_agg",  # Include Agg backend
        script_name
    ]

    # Additional options for Windows
    if sys.platform == "win32":
        cmd.extend([
            "--icon=NONE",  # No icon (or provide .ico file)
        ])

    print(f"Building executable: {exe_name}")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run PyInstaller
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("BUILD SUCCESSFUL!")
        print("=" * 60)

        # Determine the executable path
        if sys.platform == "win32":
            exe_path = f"dist/{exe_name}.exe"
        else:
            exe_path = f"dist/{exe_name}"

        print(f"\nExecutable created: {exe_path}")
        print(f"File size: {os.path.getsize(exe_path) / 1024 / 1024:.1f} MB")
        print("\nTo run the executable:")
        print(f"  ./{exe_path}" if sys.platform != "win32" else f"  {exe_path}")
        print("\nTo distribute to your friend:")
        print(f"  1. Send them the file: {exe_path}")
        print("  2. They can run it by double-clicking (Windows) or running from terminal")
        print("  3. It will download data and create plots in the same directory")
    else:
        print("\n" + "=" * 60)
        print("BUILD FAILED!")
        print("=" * 60)
        print("Check the error messages above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()