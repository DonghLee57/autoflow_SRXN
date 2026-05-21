"""
Download bulk crystal structures from the Materials Project and save as VASP POSCAR.

Target materials
----------------
mp-2574   t-ZrO2   (tetragonal zirconia, substrate)
mp-2311   NbO      (cubic, film)
mp-821    NbO2     (rutile, film)
mp-581967 Nb2O5    (film)
mp-10390  Ta2O5    (film)

Requirements
------------
    pip install mp-api pymatgen

Usage
-----
    # Interactive: prompts for API key
    python fetch_mp_structures.py

    # Non-interactive: pass key via env var or argument
    MP_API_KEY=<your_key> python fetch_mp_structures.py
    python fetch_mp_structures.py --api-key <your_key>

Outputs
-------
    structures/ZrO2_t_mp-2574.vasp
    structures/NbO_mp-2311.vasp
    structures/NbO2_mp-821.vasp
    structures/Nb2O5_mp-581967.vasp
    structures/Ta2O5_mp-10390.vasp
"""

from __future__ import annotations
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TARGETS = {
    "mp-2574":   ("ZrO2_t_mp-2574.vasp",  "t-ZrO2"),
    "mp-2311":   ("NbO_mp-2311.vasp",     "NbO"),
    "mp-821":    ("NbO2_mp-821.vasp",      "NbO2"),
    "mp-581967": ("Nb2O5_mp-581967.vasp",  "Nb2O5"),
    "mp-10390":  ("Ta2O5_mp-10390.vasp",   "Ta2O5"),
}


def get_api_key(cli_key: str | None = None) -> str:
    if cli_key:
        return cli_key
    env = os.environ.get("MP_API_KEY", "")
    if env:
        return env
    # Try reading from ~/.pmgrc.yaml (pymatgen config)
    try:
        import yaml
        pmgrc = os.path.expanduser("~/.pmgrc.yaml")
        if os.path.exists(pmgrc):
            with open(pmgrc) as f:
                data = yaml.safe_load(f) or {}
            key = data.get("PMG_MAPI_KEY", "")
            if key:
                return key
    except Exception:
        pass
    # Interactive fallback
    import getpass
    return getpass.getpass("Materials Project API key: ").strip()


def download(api_key: str, dry_run: bool = False) -> None:
    try:
        from mp_api.client import MPRester
        from pymatgen.io.vasp import Poscar
    except ImportError:
        print("Error: install mp-api and pymatgen:  pip install mp-api pymatgen")
        sys.exit(1)

    with MPRester(api_key) as mpr:
        for mp_id, (fname, label) in TARGETS.items():
            out_path = os.path.join(HERE, fname)
            print(f"  {mp_id}  {label:<10}", end="  ")
            if dry_run:
                print("[dry-run]")
                continue
            try:
                struct = mpr.get_structure_by_material_id(mp_id)
                poscar = Poscar(struct, comment=f"{label} ({mp_id})")
                poscar.write_file(out_path)
                abc = struct.lattice.abc
                angles = struct.lattice.angles
                sg = struct.get_space_group_info()
                print(
                    f"OK  {len(struct):>3} atoms  SG#{sg[1]} {sg[0]}  "
                    f"a={abc[0]:.4f} b={abc[1]:.4f} c={abc[2]:.4f} Ang  "
                    f"alpha={angles[0]:.2f} beta={angles[1]:.2f} gamma={angles[2]:.2f}"
                )
                print(f"      -> {os.path.relpath(out_path)}")
            except Exception as e:
                print(f"FAIL  {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MP structures for ZrO2/Nb-Ta-oxide interfaces")
    parser.add_argument("--api-key", default=None, help="Materials Project API key")
    parser.add_argument("--dry-run", action="store_true", help="Show targets without downloading")
    args = parser.parse_args()

    print("Materials Project structure downloader")
    print("=" * 60)
    for mp_id, (fname, label) in TARGETS.items():
        print(f"  {mp_id:<12} {label:<10} -> {fname}")
    print("=" * 60)

    if args.dry_run:
        print("Dry-run mode — no files will be written.\n")
        download("", dry_run=True)
        return

    api_key = get_api_key(args.api_key)
    if not api_key:
        print("Error: API key is required. Get one at https://materialsproject.org/api")
        sys.exit(1)

    download(api_key)
    print("\nDone. Use these filenames in config.yaml sub_path / film_path.")


if __name__ == "__main__":
    main()
