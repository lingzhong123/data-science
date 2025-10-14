#!/usr/bin/env python3
"""
Streamlit App Runner for Lazada Sales Analysis
"""

import subprocess
import sys
import os


def check_dependencies():
    """检查依赖是否安装"""
    try:
        import streamlit
        import pandas
        import plotly
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False


def main():
    """主运行函数"""
    print("🚀 Starting Lazada Sales Analysis App...")

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 运行Streamlit应用
    try:
        print("📊 Launching Streamlit application...")
        print("🌐 The app will open in your browser shortly...")
        print("🛑 To stop the app, press Ctrl+C in this terminal")

        # 运行Streamlit应用
        subprocess.run([
            "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--theme.base=light"
        ])

    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")


if __name__ == "__main__":
    main()