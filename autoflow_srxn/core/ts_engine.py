# Re-export the canonical Hessian-based TSSearcher from vibrational_analyzer.
# The NEB-based implementation that was here was dead code (never imported) and
# had a bug: it attached the calculator to individual images instead of the NEB
# object, which is incompatible with SingleCalculatorNEB.
from autoflow_srxn.vibrational_analyzer import TSSearcher

__all__ = ["TSSearcher"]
