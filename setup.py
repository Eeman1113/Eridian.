#!/usr/bin/env python3
"""
Eridian Setup Script
Automatically sets up the environment for real-time 3D Gaussian Splatting.
"""

import subprocess
import sys
import platform
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display status."""
    print(f"[Setup] {description}...")
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"[Setup] ✓ {description} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Setup] ✗ {description} failed: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("ERIDIAN - Real-time 3D Gaussian Splatting Setup")
    print("="*60 + "\n")

    system = platform.system()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    print(f"[System] OS: {system}")
    print(f"[System] Python: {python_version}")
    print(f"[System] Architecture: {platform.machine()}\n")

    # Create virtual environment if it doesn't exist
    venv_dir = Path(".venv_eridian")
    if not venv_dir.exists():
        print("[Setup] Creating virtual environment...")
        run_command(f"python3 -m venv {venv_dir}", "Virtual environment creation")
    else:
        print("[Setup] Virtual environment already exists")

    # Determine pip path
    if system == "Windows":
        pip_path = venv_dir / "Scripts" / "pip"
    else:
        pip_path = venv_dir / "bin" / "pip"

    # Install dependencies
    print("\n[Setup] Installing dependencies...")
    deps_cmd = f"{pip_path} install -r requirements.txt"
    
    # Handle PyTorch installation based on platform
    if system == "Darwin":  # macOS
        if platform.machine() == "arm64":  # Apple Silicon
            print("[Setup] Detected Apple Silicon - installing PyTorch with MPS support")
            deps_cmd = f"{pip_path} install torch torchvision"
        else:
            print("[Setup] Detected Intel Mac - installing CPU version of PyTorch")
            deps_cmd = f"{pip_path} install torch torchvision"
    elif system == "Linux":
        # Check for NVIDIA GPU
        has_nvidia = subprocess.call(
            ["which", "nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ) == 0
        
        if has_nvidia:
            print("[Setup] Detected NVIDIA GPU - installing CUDA version of PyTorch")
            deps_cmd = f"{pip_path} install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        else:
            print("[Setup] No NVIDIA GPU detected - installing CPU version of PyTorch")
            deps_cmd = f"{pip_path} install torch torchvision"
    
    run_command(deps_cmd, "Dependency installation")

    # Install remaining dependencies
    remaining_deps = "opencv-python rerun-sdk timm numpy"
    run_command(f"{pip_path} install {remaining_deps}", "Additional dependencies")

    # Create output directory
    output_dir = Path("splat")
    output_dir.mkdir(exist_ok=True)
    print(f"[Setup] Created output directory: {output_dir}")

    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    print(f"[Setup] Created config directory: {config_dir}")

    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nTo run Eridian:")
    if system == "Windows":
        print(f"  {venv_dir}\\Scripts\\python run.py")
    else:
        print(f"  {venv_dir}/bin/python run.py")
    print("\nOr simply run: python3 run.py")
    print("(It will automatically use the virtual environment)")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()