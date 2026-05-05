"""
autoflow_srxn.interface
=======================
Heteroepitaxial interface builder sub-package.

This sub-package requires **pymatgen** for the structure generation
back-end. All public symbols are exposed here so that callers can do::

    from autoflow_srxn.interface import InterfaceWorkflow

If pymatgen is not installed the import of this package succeeds but
every class raises :class:`ImportError` at instantiation time.
"""

from __future__ import annotations

try:
    from pymatgen.core import Structure  # noqa: F401 – just a probe

    from autoflow_srxn.interface.builder import (
        InterfaceCandidate,
        build_symmetric_slab,
        find_coincidences,
        get_slab_atom_count,
        get_surface_lattice_2d,
        iter_hnf_2d,
        miller_polar_inplane,
        polar_axis_for_sg,
        strain_from_F,
        POLAR_SG,
    )
    from autoflow_srxn.interface.workflow import InterfaceWorkflow
    from autoflow_srxn.interface.visualization import (
        save_candidates_json,
        save_candidates_html,
    )

    _HAS_PYMATGEN = True

except ImportError as _err:

    _HAS_PYMATGEN = False

    class _MissingDependency:  # type: ignore[no-redef]
        """Placeholder that raises ImportError on instantiation."""

        _msg = (
            "autoflow_srxn.interface requires pymatgen.\n"
            f"(Original error: {_err})"
        )

        def __init__(self, *a, **kws):
            raise ImportError(self._msg)

        @classmethod
        def __init_subclass__(cls, **kws):
            pass

    # Expose stub symbols so that ``from autoflow_srxn.interface import X``
    # doesn't raise NameError at import time.
    InterfaceWorkflow = _MissingDependency  # type: ignore[misc,assignment]
    InterfaceCandidate = _MissingDependency  # type: ignore[misc,assignment]
    build_symmetric_slab = _MissingDependency  # type: ignore[assignment]
    get_surface_lattice_2d = _MissingDependency  # type: ignore[assignment]
    get_slab_atom_count = _MissingDependency  # type: ignore[assignment]
    find_coincidences = _MissingDependency  # type: ignore[assignment]
    iter_hnf_2d = _MissingDependency  # type: ignore[assignment]
    strain_from_F = _MissingDependency  # type: ignore[assignment]
    polar_axis_for_sg = _MissingDependency  # type: ignore[assignment]
    miller_polar_inplane = _MissingDependency  # type: ignore[assignment]
    save_candidates_json = _MissingDependency
    save_candidates_html = _MissingDependency
    POLAR_SG = set()  # type: ignore[assignment]

__all__ = [
    "InterfaceWorkflow",
    "InterfaceCandidate",
    "build_symmetric_slab",
    "get_surface_lattice_2d",
    "get_slab_atom_count",
    "find_coincidences",
    "iter_hnf_2d",
    "strain_from_F",
    "polar_axis_for_sg",
    "miller_polar_inplane",
    "save_candidates_json",
    "save_candidates_html",
    "POLAR_SG",
    "_HAS_PYMATGEN",
]
