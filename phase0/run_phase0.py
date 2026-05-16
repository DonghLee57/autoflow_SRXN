"""Phase 0 master runner.

Steps (in order):
  1. Build Ni(PF3)4 geometry              → structures/NiPF3_4.vasp
  2. Relax bulk Si & SiO2 (cell relax)    → structures/{Si,SiO2}_relaxed.vasp
  3. Relax molecules (mode-following)     → structures/{AllylCpNi,inhibitor,NiPF3_4}_relaxed.vasp
  4. Validate AllylCpNi haptic code       → console report

Usage:
  python phase0/run_phase0.py [step1] [step2] ...

  Available steps: build_nipf3, bulk, molecules, validate
  No args → run all steps.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

STEPS = {
    "build_nipf3": {
        "desc": "Build Ni(PF3)4 structure",
        "script": ROOT / "phase0/build_NiPF3_4.py",
    },
    "bulk": {
        "desc": "Bulk cell relaxation (Si, SiO2)",
        "script": ROOT / "phase0/relax_bulk.py",
    },
    "molecules": {
        "desc": "Mode-following relaxation (AllylCpNi, inhibitor, NiPF3_4)",
        "script": ROOT / "phase0/relax_molecules.py",
    },
    "validate": {
        "desc": "AllylCpNi haptic code validation",
        "script": ROOT / "phase0/validate_allylcpni.py",
    },
}


def run_step(name, info):
    print("\n" + "#" * 70)
    print(f"#  PHASE 0 STEP: {name.upper()}")
    print(f"#  {info['desc']}")
    print("#" * 70)

    cmd = [sys.executable, str(info["script"])]
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{name}' failed with exit code {result.returncode}")
        return False
    return True


def main():
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(STEPS.keys())
    unknown = [s for s in requested if s not in STEPS]
    if unknown:
        print(f"Unknown steps: {unknown}. Valid: {list(STEPS.keys())}")
        sys.exit(1)

    failed = []
    for name in requested:
        ok = run_step(name, STEPS[name])
        if not ok:
            failed.append(name)

    print("\n" + "=" * 70)
    print("PHASE 0 COMPLETE")
    print("=" * 70)
    done = [n for n in requested if n not in failed]
    if done:
        print(f"  Succeeded: {done}")
    if failed:
        print(f"  Failed:    {failed}")
        sys.exit(1)
    else:
        print("  All steps passed.")


if __name__ == "__main__":
    main()
