"""
build.py — One-click build script for CVFrame_v3.

Usage (from the build/ folder or project root):
    python build/build.py

Outputs:
    Windows : build/dist/CVFrame_v3.exe
    macOS   : build/dist/CVFrame_v3.app

Requirements:
    pip install pyinstaller
"""

import subprocess
import sys
import shutil
import os

# Always resolve paths relative to this script, regardless of where it's called from
BUILD_DIR = os.path.dirname(os.path.abspath(__file__))


def clean():
    """Remove previous build artefacts."""
    for folder in ("build", "dist"):
        full = os.path.join(BUILD_DIR, folder)
        if os.path.exists(full):
            print(f"Cleaning {full}/")
            shutil.rmtree(full)


def build():
    print("=" * 50)
    print(f"Building CVFrame_v3  ({sys.platform})")
    print("=" * 50)

    # Ensure PyInstaller is available
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    clean()

    spec_file = os.path.join(BUILD_DIR, "CVFrame2.spec")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", spec_file,
         "--distpath", os.path.join(BUILD_DIR, "dist"),
         "--workpath", os.path.join(BUILD_DIR, "build")],
        check=False,
    )

    if result.returncode == 0:
        print("\n✅ Build succeeded!")
        if sys.platform == "darwin":
            print(f"   Output: {os.path.join(BUILD_DIR, 'dist', 'CVFrame_v3.app')}")
        else:
            print(f"   Output: {os.path.join(BUILD_DIR, 'dist', 'CVFrame_v3.exe')}")
    else:
        print("\n❌ Build failed. Check the output above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    build()
