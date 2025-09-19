# Building Windows Executable from Mac

## Option 1: GitHub Actions (Free & Automatic)

1. Create a GitHub repository and push your code:
```bash
git init
git add shiller_regime_plot_standalone.py
git add .github/workflows/build-windows.yml
git commit -m "Add Windows build workflow"
git remote add origin https://github.com/YOUR_USERNAME/shiller-regime-plot
git push -u origin main
```

2. Go to your repo on GitHub → Actions tab
3. The build will run automatically
4. Download the .exe from Artifacts

## Option 2: AWS EC2 (Pay-per-use)

1. Launch a Windows instance on AWS EC2:
   - AMI: Microsoft Windows Server 2022 Base
   - Instance type: t2.micro (free tier eligible)
   - Stop instance after 1 hour = ~$0.02

2. Connect via RDP and run:
```powershell
# Install Python
winget install Python.Python.3.10

# Install dependencies
pip install requests pandas numpy matplotlib xlrd==1.2.0 pyinstaller

# Build executable
python build_exe.py
```

3. Download the .exe file via S3 or email

## Option 3: Azure DevOps (Free)

1. Create free Azure DevOps account
2. Create a pipeline with windows-latest agent
3. Use similar steps as GitHub Actions
4. Download artifact

## Option 4: Use a Friend's Windows PC

Send them:
- `shiller_regime_plot_standalone.py`
- `build_exe.py`
- `requirements_exe.txt`

Have them run:
```cmd
pip install -r requirements_exe.txt
python build_exe.py
```

## Option 5: Wine on Mac (Unreliable)

```bash
# Install Wine
brew install --cask wine-stable

# Download Python Windows installer
curl -O https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe

# Install Python in Wine
wine64 python-3.10.11-amd64.exe

# Install packages
wine64 pip install requests pandas numpy matplotlib xlrd==1.2.0 pyinstaller

# Build
wine64 pyinstaller --onefile --console --name=ShillerRegimePlot shiller_regime_plot_standalone.py
```

Note: Wine approach often fails due to compatibility issues, especially on M1/M2 Macs.