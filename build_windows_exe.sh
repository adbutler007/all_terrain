#!/bin/bash
# Build Windows executable using Docker

echo "============================================================"
echo "Building Windows Executable using Docker"
echo "============================================================"
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker Desktop first:"
    echo "https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "Pulling PyInstaller Windows Docker image..."
docker pull cdrx/pyinstaller-windows:python3

echo
echo "Building Windows executable..."

# Run the Docker container to build the Windows exe
docker run --rm \
    -v "$(pwd):/src" \
    cdrx/pyinstaller-windows:python3 \
    "pip install requests pandas numpy matplotlib xlrd==1.2.0 && \
     pyinstaller --onefile \
                 --console \
                 --name=ShillerRegimePlot \
                 --clean \
                 --noconfirm \
                 --hidden-import=matplotlib.backends.backend_pdf \
                 --hidden-import=matplotlib.backends.backend_agg \
                 --hidden-import=PIL \
                 --hidden-import=PIL._imaging \
                 shiller_regime_plot_standalone.py"

# Check if build was successful
if [ -f "dist/ShillerRegimePlot.exe" ]; then
    echo
    echo "============================================================"
    echo "BUILD SUCCESSFUL!"
    echo "============================================================"
    echo
    echo "Windows executable created: dist/ShillerRegimePlot.exe"
    echo "File size: $(ls -lh dist/ShillerRegimePlot.exe | awk '{print $5}')"
    echo
    echo "To distribute to Windows users:"
    echo "  1. Send them the file: dist/ShillerRegimePlot.exe"
    echo "  2. They can double-click to run it"
    echo "  3. It will download data and create plots in the same directory"
else
    echo
    echo "============================================================"
    echo "BUILD FAILED!"
    echo "============================================================"
    echo "Check the error messages above for details."
    exit 1
fi