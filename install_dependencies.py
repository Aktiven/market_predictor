# install_dependencies.py
import subprocess
import sys
import os


def install_requirements():
    """Install requirements with error handling"""
    requirements = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "yfinance>=0.2.18",
        "requests>=2.31.0",
        "tensorflow>=2.13.0",
        "streamlit>=1.28.0",
        "textblob>=0.17.1",
        "vaderSentiment>=3.3.2",
        "newspaper3k>=0.2.8",
        "beautifulsoup4>=4.12.0",
        "pandas-ta>=0.3.14",
        "arch>=6.2.0",
        "statsmodels>=0.14.0",
        "python-dateutil>=2.8.0",
        "tzlocal>=4.3.0"
    ]

    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            # Try without version specifier
            base_package = package.split('>=')[0].split('==')[0]
            try:
                print(f"Trying to install {base_package} without version constraint...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", base_package])
                print(f"✓ Successfully installed {base_package}")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {base_package}")


if __name__ == "__main__":
    install_requirements()
    print("\nInstallation completed! You can now run: streamlit run app.py")