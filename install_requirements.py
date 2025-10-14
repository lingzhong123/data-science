import subprocess
import sys


def install_packages():
    required_packages = [
        "streamlit==1.28.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "plotly==5.15.0",
        "scikit-learn==1.3.0",
        "joblib==1.3.2",
        "openpyxl==3.1.2"
    ]

    print("Installing required packages...")
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

    print("\n🎉 All packages installed successfully!")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    install_packages()