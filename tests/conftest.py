import sys
import os

# Add project root so `api`, `aggregation`, etc. are importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

# If running outside the venv, pull in the venv's site-packages so that
# installed dependencies (fastapi, etc.) are available.
_venv_site = os.path.join(_project_root, ".venv", "lib", "python3.13", "site-packages")
if os.path.isdir(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)
    # Evict any already-cached packages so they reload from the venv.
    # typing_extensions is imported by pytest before conftest runs; the system
    # version is too old for pydantic-core (missing `Sentinel`).
    for _mod in list(sys.modules):
        if _mod == "typing_extensions" or _mod.startswith("pydantic"):
            del sys.modules[_mod]
