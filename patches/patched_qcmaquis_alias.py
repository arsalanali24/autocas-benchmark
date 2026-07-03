#!/usr/bin/env python3
"""
patched_qcmaquis_alias.py

Fixes a third confirmed bug -- a version-skew naming mismatch between
scine_autocas 3.0.0 and the installed scine_qcmaquis 4.0.0, NOT a
missing or broken QCMaquis installation.

THE PROBLEM:
  scine_autocas/interfaces/pyscf/pyscf.py, _setup_qcmaquis(), does:
      qcmaquis = pyscf_interface.QcMaquis(self.pyscf_mol)
  but the installed scine_qcmaquis.pyscf_interface.pyscf_interface
  module has no class named QcMaquis -- the actual class is named
  DMRGSolver. This produced:
      AttributeError: module 'scine_qcmaquis.pyscf_interface.pyscf_interface'
      has no attribute 'QcMaquis'
  three sub-CASs into the FIRST successful run past both previously
  fixed bugs (UHF/ECP in PyscfInterface, infinite loop in
  LargeSpaces._partition_space) -- confirming the actual DMRG engine
  (the compiled _dmrg extension module) imports and is present; this is
  purely a class-name mismatch one layer up.

WHY THE ALIAS IS SAFE (verified, not assumed): every method and
attribute _setup_qcmaquis touches on the constructed object was checked
directly against the real DMRGSolver/ParametersWrapper source:
  - qcmaquis.parameters                  -> DMRGSolver.__init__ sets
                                             self.parameters = ParametersWrapper()
  - .parameters.set_entropies()          -> ParametersWrapper.set_entropies()
  - .parameters.set(name, value)         -> ParametersWrapper.set()
  - .parameters.set_result_path(path)    -> ParametersWrapper.set_result_path()
  - .parameters.set_checkpoint_path(path)-> ParametersWrapper.set_checkpoint_path()
  - qcmaquis.fiedler = bool              -> DMRGSolver.__init__ sets
                                             self.fiedler: bool (plain attribute)
  - qcmaquis.file_path = path_or_None    -> DMRGSolver.__init__ sets
                                             self.file_path: str (plain attribute)
  - QcMaquis(mol)                        -> DMRGSolver.__init__(self, mol,
                                             verbose=None, fiedler=False, **kwargs)
                                             accepts mol positionally

Every call site matches. This is a complete, not partial, fix.

Usage: call apply_patch() before the workflow runs, alongside the
PyscfInterface and LargeSpaces patches (same monkeypatch pattern,
same timing requirement -- before _setup_qcmaquis is actually called).
"""


def apply_patch():
    """Alias QcMaquis -> DMRGSolver in the pyscf_interface module
    namespace, so scine_autocas's existing, untouched call to
    `pyscf_interface.QcMaquis(...)` resolves correctly."""
    from scine_qcmaquis.pyscf_interface import pyscf_interface
    from scine_qcmaquis.pyscf_interface.pyscf_interface import DMRGSolver

    if hasattr(pyscf_interface, "QcMaquis"):
        print(
            f"[patch] pyscf_interface.QcMaquis already exists "
            f"({pyscf_interface.QcMaquis}) -- not overwriting, alias not needed."
        )
        return

    pyscf_interface.QcMaquis = DMRGSolver
    print(f"[patch] scine_qcmaquis.pyscf_interface.pyscf_interface.QcMaquis -> {DMRGSolver}")
    assert pyscf_interface.QcMaquis is DMRGSolver, "Monkeypatch did not take"


if __name__ == "__main__":
    # Standalone smoke test: confirm the alias resolves and the
    # resulting object's interface actually works for every call
    # _setup_qcmaquis makes, using a minimal dummy mol-like object
    # (DMRGSolver.__init__ only reads mol.verbose from it).
    apply_patch()

    from scine_qcmaquis.pyscf_interface import pyscf_interface

    class _DummyMol:
        verbose = 3

    print()
    print("=== Constructing pyscf_interface.QcMaquis(dummy_mol) ===")
    qcmaquis = pyscf_interface.QcMaquis(_DummyMol())
    print(f"Constructed: {qcmaquis} (type: {type(qcmaquis).__name__})")

    print()
    print("=== Exercising every call _setup_qcmaquis makes ===")
    qcmaquis.parameters.set_entropies()
    print("  parameters.set_entropies() OK")
    qcmaquis.parameters.set("nsweeps", 5)
    print("  parameters.set('nsweeps', 5) OK")
    qcmaquis.parameters.set("max_bond_dimension", 250)
    print("  parameters.set('max_bond_dimension', 250) OK")
    qcmaquis.fiedler = True
    print(f"  qcmaquis.fiedler = True OK (now: {qcmaquis.fiedler})")
    qcmaquis.file_path = "/tmp/test_path"
    print(f"  qcmaquis.file_path = ... OK (now: {qcmaquis.file_path})")
    qcmaquis.parameters.set_result_path("results_test")
    print(f"  parameters.set_result_path(...) OK (now: {qcmaquis.parameters.get_result_path()})")
    qcmaquis.parameters.set_checkpoint_path("checkpoint_test")
    print(f"  parameters.set_checkpoint_path(...) OK (now: {qcmaquis.parameters.get_checkpoint_path()})")

    print()
    print("All calls succeeded -- the alias is a complete, working fix.")
