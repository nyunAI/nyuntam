from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parent  # cli directory
REPO_ROOT = ROOT.parent  # repository root
NYUNTAM = REPO_ROOT / "nyuntam"  # path to nyuntam directory
CLI = REPO_ROOT / "cli"  # path to cli directory
SERVICES = REPO_ROOT / "services"  # path to services directory

__all__ = ["NYUNTAM", "REPO_ROOT", "CLI", "SERVICES"]
