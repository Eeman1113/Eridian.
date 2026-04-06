#!/usr/bin/env python3
"""
Eridian - Real-time monocular 3D point cloud reconstruction.

Compatibility shim so that `python main.py` and `from main import ...`
keep working when running from the cloned repo.
"""
from eridian.main import *          # noqa: F401,F403
from eridian.main import cli_main   # noqa: F401

if __name__ == "__main__":
    cli_main()
