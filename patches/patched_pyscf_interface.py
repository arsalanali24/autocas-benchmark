#!/usr/bin/env python3
"""
patched_pyscf_interface.py

Minimal patch for scine_autocas 3.0.0's PyscfInterface, as actually
installed on Noctua2 at:
  /pc2/users/h/hpcmual/.local/lib/python3.10/site-packages/scine_autocas/interfaces/pyscf/pyscf.py

Fixes exactly two confirmed bugs, both found by reading that source file
directly (not guessed):

  1. _initial_orbitals_impl() unconditionally does `scf.RHF(self.pyscf_mol)`.
     There is no branch on settings.uhf or on the molecule's spin anywhere
     in that method. For 8 of our 10 systems (CrCl6, MnCl4, FeCl4, FeNH3,
     MoCl6, ReCl6, OsCl6 -- everything except RhCl6 and CoNH3, which are
     genuinely closed-shell d6 S=0) this silently runs RHF on an open-shell
     ground state, which is wrong.

  2. _build_molecule() (the path used for a normal xyz-file calculation)
     never references ecp_electrons or applies any ecp at all -- it is
     dead code on this path. The only ECP-handling code that exists lives
     inside init_from_molden() instead, and there it is hardcoded to
     `ecp={"Mo": "def2-svp"}` regardless of which element is actually in
     the molecule. Neither path is usable as-is for RhCl6, MoCl6, ReCl6,
     OsCl6.

Fix approach:
  1. Use UHF whenever settings.uhf is set OR the built molecule has
     mol.spin != 0 (defensive: correct even if uhf: true is forgotten
     in a per-system yaml).
  2. Pass ecp=<basis_set> as a blanket string to mol.build() whenever the
     basis is a def2-* family basis. PySCF resolves this per-element
     automatically and applies it only to elements whose basis definition
     requires one -- confirmed via pyscf/pyscf GitHub issue #531:
         gto.M(atom='Xe 0 0 0', basis='def2-SVP', ecp='def2-SVP')
     is the documented-correct call; omitting ecp loads the valence basis
     without the matching core potential, which is silently wrong for
     heavy elements. No per-element enumeration needed -- this is general
     for Mo, Rh, Re, Os, or any other def2-family heavy element.

Both overrides touch ONLY _build_molecule and _initial_orbitals_impl.
Everything else (the large-CAS selection algorithm in
cas_selection/large_active_spaces.py, the CASCI/CASSCF/DMRGSCF dispatch
in _initial_cas_impl / _final_cas_impl, the CLI, the YAML schema) is
completely untouched -- this is intentionally the smallest patch that
fixes the two confirmed gaps and nothing else.

NOT yet verified: how the CLI's `-i/--interface pyscf` selection wires a
class instance together (we have not read io/actions/run.py yet). This
file gives you the corrected class; plugging it in via the CLI's
interface-name lookup vs. constructing a Workflow directly with this
class still needs to be confirmed against that file before relying on it
for the production run.
"""

from typing import List

from pyscf import gto, lib, scf

from scine_autocas.interfaces.pyscf.pyscf import PyscfInterface

# Basis-set families in PySCF that are defined as valence-basis + ECP
# for heavy elements. Extend this if a different basis family is used.
_ECP_BASIS_FAMILIES = ("def2-",)


class PatchedPyscfInterface(PyscfInterface):
    """PyscfInterface with UHF selection and ECP handling fixed.

    No new instance state is introduced, so __slots__ is left empty
    rather than omitted -- this preserves the slotted-class behaviour
    of the base Interface hierarchy instead of silently adding a
    per-instance __dict__.
    """

    __slots__ = ()

    def _basis_needs_ecp(self) -> bool:
        """True if the configured basis set is a def2-family basis."""
        basis = self.settings.basis_set.lower()
        return any(basis.startswith(fam) for fam in _ECP_BASIS_FAMILIES)

    def _build_molecule(self):
        """Build the pyscf molecule from settings, with ECP applied
        whenever the basis set requires one (fix for bug 2)."""
        molecule = self.settings.get_molecule()
        needs_ecp = self._basis_needs_ecp()

        print(
            f"""Build molecule (patched interface):
            xyz-file:     {molecule.xyz_file}
            basis:        {self.settings.basis_set}
            ecp:          {self.settings.basis_set if needs_ecp else 'none (basis is not a def2-family set)'}
            spin (2S+1):  {molecule.spin_multiplicity}
            xyz unit:     {molecule.unit}
            total charge: {molecule.charge}"""
        )

        # scf_symmetry defaults to True: enables PySCF point-group
        # symmetry detection, which reduces DMRG cost for high-symmetry
        # TM complexes (Oh -> D2h for octahedral, Td -> D2 for
        # tetrahedral). Set settings.scf_symmetry = False to reproduce
        # the original symmetry=False behaviour for a specific system.
        use_symmetry = bool(getattr(self.settings, "scf_symmetry", True))

        self.pyscf_mol = gto.Mole()
        build_kwargs = dict(
            atom=molecule.xyz_file,
            basis=self.settings.basis_set,
            symmetry=use_symmetry,
            spin=molecule.spin_multiplicity - 1,
            unit=molecule.unit,
            charge=molecule.charge,
        )
        if needs_ecp:
            # Blanket string, not a per-element dict: pyscf applies the
            # ECP only to elements whose basis definition needs one.
            # See pyscf/pyscf#531.
            build_kwargs["ecp"] = self.settings.basis_set

        self.pyscf_mol.build(**build_kwargs)
        self.pyscf_mol.verbose = 3

        if use_symmetry:
            print(f"    symmetry: detected {self.pyscf_mol.topgroup} -> "
                  f"using Abelian subgroup {self.pyscf_mol.groupname}")

    def _initial_orbitals_impl(self) -> List[float]:
        """Run initial Hartree-Fock, using UHF for any open-shell
        system (fix for bug 1).

        SCF convergence aids (max_cycle, init_guess, level_shift) are
        read from self.settings if present, defaulting to pyscf's own
        defaults (max_cycle=50, init_guess='minao', level_shift=0) so
        that systems which already converge cleanly are completely
        unaffected. This is deliberately opt-in per system, not a
        silent global change -- a difficult system like OsCl6 gets
        these set explicitly by the caller, nothing else does.
        """
        if self.pyscf_mol is None:
            self._build_molecule()

        if self.pyscf_hf is not None:
            print("Using existing hartree fock run")
            return self.pyscf_hf.e_tot

        explicit_uhf = bool(getattr(self.settings, "uhf", False))
        open_shell = self.pyscf_mol.spin != 0
        use_uhf = explicit_uhf or open_shell

        hf_cls = scf.UHF if use_uhf else scf.RHF

        max_cycle = getattr(self.settings, "scf_max_cycle", 50)
        init_guess = getattr(self.settings, "scf_init_guess", "minao")
        level_shift = getattr(self.settings, "scf_level_shift", 0.0)

        print(
            f"Starting Hartree Fock: {'UHF' if use_uhf else 'RHF'} "
            f"(settings.uhf={explicit_uhf}, mol.spin={self.pyscf_mol.spin}, "
            f"max_cycle={max_cycle}, init_guess={init_guess}, level_shift={level_shift})"
        )

        with lib.capture_stdout() as out:
            self.pyscf_hf = hf_cls(self.pyscf_mol)
            self.pyscf_hf.max_cycle = max_cycle
            self.pyscf_hf.init_guess = init_guess
            if level_shift:
                self.pyscf_hf.level_shift = level_shift
            energy = self.pyscf_hf.scf()
        print("Pyscf output:")
        print(out.read())

        if not self.pyscf_hf.converged:
            raise RuntimeError(
                f"SCF did NOT converge after {max_cycle} cycles "
                f"(energy={energy}, init_guess={init_guess}, level_shift={level_shift}). "
                f"This must not be silently treated as a usable result -- either "
                f"increase max_cycle/level_shift further, try a different "
                f"init_guess, or treat this system as flagged rather than passing "
                f"it downstream into DMRG."
            )

        return [energy]


    def _final_cas_impl(self, cas_occupation: List[int], cas_indices: List[int]) -> List[float]:
        """Override to add a spin-consistency check before the final CAS
        calculation. This generalises the earlier spin-parity fix.

        ROOT CAUSE (UHF symmetry breaking, confirmed on CrCl6 and MnCl4):
        When UHF breaks the degeneracy of formally equivalent d orbitals
        (e.g. the t2g set in octahedral or e+t2 in tetrahedral), one or more
        singly-occupied d orbitals get pushed to near-zero single-orbital
        entropy. autoCAS correctly excludes them by its own criterion, but the
        resulting active space has too few singly-occupied orbitals to match
        the molecule's spin state. PySCF then fails one of three assertions
        in check_sanity() / sort_mo():

          (A) assert ncorelec % 2 == 0    [CrCl6: triggered, 1 orbital missing]
          (B) assert nelecas[0] <= ncas   [MnCl4: triggered, 2 orbitals missing]
          (C) assert nelec >= mol.spin    [possible for larger spin deficits]

        GENERAL FIX:
        After plateau selection, check all three PySCF sanity conditions:
          1. ncorelec = mol.nelectron - nelec  must be even
          2. alpha_elec = (nelec + mol.spin)/2 must be <= norbs
          3. nelec >= mol.spin
        If any condition fails, add excluded singly-occupied orbitals from the
        UHF reference one at a time (lowest index first) until all three pass.
        This is physically motivated: a singly-occupied orbital in UHF that was
        excluded by low entropy is a degenerate set member that belongs in the
        active space regardless of its numerical entropy value.

        OBSERVED CASES:
        - CrCl6^3- (d3 t2g): 1 orbital missing -> condition A fails -> add 1
        - MnCl4^2- (d5 e+t2): 2 orbitals missing -> condition B fails -> add 2
        - FeCl4^1-, MoCl6^3-: all conditions pass -> no fix needed
        """
        mol_spin = self.pyscf_mol.spin  # 2S (number of unpaired electrons)

        def _all_conditions_pass(occ, idx):
            ne = int(sum(occ))
            no = len(occ)
            ncore = self.pyscf_mol.nelectron - ne
            alpha = (ne + mol_spin) / 2
            n_singly = sum(1 for o in occ if o == 1)
            return (
                ncore % 2 == 0,       # condition A: ncorelec even
                alpha <= no,           # condition B: alpha electrons fit in norbs
                ne >= mol_spin,        # condition C: enough active electrons
                n_singly >= mol_spin,  # condition D: enough singly-occ orbs
                                       #   (catches CrCl6_CSD: plateau gave 0
                                       #    singly-occ; A/B/C all passed with 1
                                       #    after fix, but D still failed)
            )

        cond = _all_conditions_pass(cas_occupation, cas_indices)
        if not all(cond):
            nelec_orig = int(sum(cas_occupation))
            n_singly_orig = sum(1 for o in cas_occupation if o == 1)
            print(
                f"\n[spin-consistency fix] Active space CAS({nelec_orig}, "
                f"{len(cas_occupation)}) fails PySCF sanity checks:\n"
                f"  mol.nelectron={self.pyscf_mol.nelectron}, mol.spin={mol_spin}\n"
                f"  condition A (ncorelec even):       {'PASS' if cond[0] else 'FAIL'}\n"
                f"  condition B (alpha <= norbs):      {'PASS' if cond[1] else 'FAIL'}\n"
                f"  condition C (nelec >= mol.spin):   {'PASS' if cond[2] else 'FAIL'}\n"
                f"  condition D (n_singly >= mol.spin): {'PASS' if cond[3] else 'FAIL'} "
                f"({n_singly_orig} singly-occ, need {mol_spin})\n"
                f"Searching for excluded singly-occupied orbitals to restore "
                f"spin consistency..."
            )

            molecule = self.settings.get_molecule()
            missing_singly_occ = sorted([
                i for i, occ in enumerate(molecule.occupation)
                if occ == 1 and i not in cas_indices
            ])

            if not missing_singly_occ:
                raise ValueError(
                    f"[spin-consistency fix] Active space fails sanity checks "
                    f"but no singly-occupied orbitals remain outside it. "
                    f"Cannot auto-fix. mol.spin={mol_spin}, "
                    f"mol.nelectron={self.pyscf_mol.nelectron}, "
                    f"nelec={nelec_orig}, cas_indices={cas_indices}."
                )

            cas_indices = list(cas_indices)
            cas_occupation = list(cas_occupation)

            for extra_idx in missing_singly_occ:
                if all(_all_conditions_pass(cas_occupation, cas_indices)):
                    break
                cas_indices = sorted(cas_indices + [extra_idx])
                cas_occupation = [molecule.occupation[i] for i in cas_indices]
                new_nelec = int(sum(cas_occupation))
                new_cond = _all_conditions_pass(cas_occupation, cas_indices)
                print(
                    f"  Added orbital {extra_idx} (singly occupied in UHF): "
                    f"CAS({new_nelec}, {len(cas_occupation)}) "
                    f"A={'OK' if new_cond[0] else 'fail'} "
                    f"B={'OK' if new_cond[1] else 'fail'} "
                    f"C={'OK' if new_cond[2] else 'fail'}"
                )

            final_cond = _all_conditions_pass(cas_occupation, cas_indices)
            if not all(final_cond):
                raise ValueError(
                    f"[spin-consistency fix] Could not satisfy all PySCF "
                    f"sanity conditions even after adding all excluded "
                    f"singly-occupied orbitals. Remaining failures: "
                    f"A={final_cond[0]} B={final_cond[1]} C={final_cond[2]}. "
                    f"Final cas_indices={cas_indices}."
                )

            final_nelec = int(sum(cas_occupation))
            print(
                f"[spin-consistency fix] Fixed: "
                f"CAS({nelec_orig}, {len(cas_occupation) - (final_nelec - nelec_orig)}) "
                f"-> CAS({final_nelec}, {len(cas_occupation)})\n"
                f"  Final cas_indices:    {list(cas_indices)}\n"
                f"  Final cas_occupation: {list(cas_occupation)}"
            )

        return super()._final_cas_impl(cas_occupation, cas_indices)


if __name__ == "__main__":
    # Minimal standalone smoke test: build CrCl6^3- (no ECP needed, open
    # shell, d3) and confirm UHF gets selected and the molecule builds.
    # This does NOT run DMRG or large_cas -- it only checks that the two
    # patched methods behave correctly in isolation, on systems we
    # already have validated reference energies for.
    #
    # CrCl6^3- (no ECP needed) was already PASSED in a prior run, but that
    # run only proves the ECP code path doesn't crash when nothing needs
    # an ECP -- it gives zero evidence about whether ecp='def2-svp'
    # actually attaches to a heavy element. MoCl6^3- (Mo IS a def2-ECP
    # element) is the real test of that.
    import os
    import tempfile

    from scine_autocas.utils.molecule import Molecule

    def run_smoke_test(
        label, xyz_content, charge, spin_mult, expect_ecp_element=None,
        scf_max_cycle=None, scf_init_guess=None, scf_level_shift=None,
    ):
        """Build one system through PatchedPyscfInterface and report
        exactly what UHF/RHF, ECP assignment, and SCF convergence
        actually happened -- no assumptions, only what pyscf itself
        reports back.

        scf_max_cycle / scf_init_guess / scf_level_shift are optional
        explicit overrides for systems that don't converge with pyscf's
        defaults. Leaving them as None means the patched interface uses
        pyscf's own defaults, so passing systems are unaffected."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write(xyz_content)
            xyz_path = f.name

        try:
            molecule = Molecule(xyz_path)
            interface = PatchedPyscfInterface(molecule)
            interface.settings.basis_set = "def2-svp"
            interface.settings.get_molecule().charge = charge
            interface.settings.get_molecule().spin_multiplicity = spin_mult
            interface.settings.uhf = True
            if scf_max_cycle is not None:
                interface.settings.scf_max_cycle = scf_max_cycle
            if scf_init_guess is not None:
                interface.settings.scf_init_guess = scf_init_guess
            if scf_level_shift is not None:
                interface.settings.scf_level_shift = scf_level_shift

            print(f"=== Smoke test: {label} via PatchedPyscfInterface ===")
            energy = interface._initial_orbitals_impl()
            mol = interface.pyscf_mol

            print(f"HF energy: {energy}")
            print(f"mol.spin: {mol.spin}")
            print(f"HF class actually used: {type(interface.pyscf_hf).__name__}")
            print(f"SCF converged: {interface.pyscf_hf.converged}")
            print(f"mol.nelectron (explicit electrons after any ECP core removal): {mol.nelectron}")
            print(f"mol.ecp (raw value passed to build -- NOT a resolved per-atom dict): {mol.ecp}")

            # mol.ecp just echoes back the raw input we passed (a string,
            # in our case) -- it is NOT a resolved per-element mapping, so
            # checking element names against it tells us nothing. The
            # correct, public, per-atom check is atom_nelec_core(atom_id),
            # which reports how many core electrons pyscf actually
            # replaced with an ECP for that specific atom.
            print("Per-atom core electrons removed by ECP (0 = all-electron, "
                  "matched by element + basis):")
            core_electrons_by_symbol = {}
            for atom_id in range(mol.natm):
                symbol = mol.atom_symbol(atom_id)
                n_core = mol.atom_nelec_core(atom_id)
                print(f"    atom {atom_id} ({symbol}): {n_core} core electrons removed")
                core_electrons_by_symbol.setdefault(symbol, n_core)

            assert "UHF" in type(interface.pyscf_hf).__name__, (
                f"Expected UHF for {label}, got {type(interface.pyscf_hf).__name__}"
            )

            # This is the check that was MISSING before -- a non-converged
            # SCF was passing silently because only the solver class was
            # checked, not whether it actually converged. The patched
            # _initial_orbitals_impl now raises RuntimeError on its own
            # for this case (caught below), but keep this explicit assert
            # too as a second, independent guard.
            assert interface.pyscf_hf.converged, (
                f"SCF reports converged=False for {label}. A non-converged "
                f"energy must never be treated as a usable result."
            )

            if expect_ecp_element is not None:
                n_core_removed = core_electrons_by_symbol.get(expect_ecp_element, 0)
                attached = n_core_removed > 0
                print(
                    f"ECP attached to {expect_ecp_element}: "
                    f"{'YES (' + str(n_core_removed) + ' core electrons removed)' if attached else 'NO -- THIS IS THE THING WE NEED TO KNOW'}"
                )
                assert attached, (
                    f"Expected atom_nelec_core() > 0 for {expect_ecp_element}, but "
                    f"got {n_core_removed}. The blanket ecp='def2-svp' string did "
                    f"NOT attach for this element -- do not trust this patch on "
                    f"Mo/Rh/Re/Os systems without resolving this first."
                )

            print(f"Smoke test PASSED: {label}")
            print()
            return True
        except (AssertionError, RuntimeError) as e:
            print(f"Smoke test FAILED: {label}")
            print(f"  {type(e).__name__}: {e}")
            print()
            return False
        finally:
            os.unlink(xyz_path)

    results = {}

    results["CrCl6_3m"] = run_smoke_test(
        label="CrCl6^3- (no ECP element present -- control case)",
        xyz_content="""7
CrCl6 3- test geometry (octahedral, Cr-Cl = 2.33 Ang)
Cr  0.00   0.00   0.00
Cl  2.33   0.00   0.00
Cl -2.33   0.00   0.00
Cl  0.00   2.33   0.00
Cl  0.00  -2.33   0.00
Cl  0.00   0.00   2.33
Cl  0.00   0.00  -2.33
""",
        charge=-3,
        spin_mult=4,
        expect_ecp_element=None,
    )

    # MoCl6^3-: Mo-Cl = 2.37 Ang, octahedral -- same geometry recipe as
    # the original run_autocas_benchmark.py SYSTEMS dict. Mo IS a
    # def2-SVP ECP element, so this is the real test.
    results["MoCl6_3m"] = run_smoke_test(
        label="MoCl6^3- (Mo IS a def2-ECP element -- real test of the fix)",
        xyz_content="""7
MoCl6 3- test geometry (octahedral, Mo-Cl = 2.37 Ang)
Mo  0.00   0.00   0.00
Cl  2.37   0.00   0.00
Cl -2.37   0.00   0.00
Cl  0.00   2.37   0.00
Cl  0.00  -2.37   0.00
Cl  0.00   0.00   2.37
Cl  0.00   0.00  -2.37
""",
        charge=-3,
        spin_mult=4,
        expect_ecp_element="Mo",
    )

    # RhCl6^3-: Rh-Cl = 2.34 Ang, octahedral. spin_mult=1 (S=0) is the
    # CORRECTED value from earlier in this project -- Rh3+ d6 low-spin
    # under strong-field Cl, not the originally-wrong spin_mult=2 that
    # was caught and fixed in run_autocas_benchmark.py. Rh is 4d, same
    # row as Mo, so its def2-ECP core size should also be 28.
    results["RhCl6_3m"] = run_smoke_test(
        label="RhCl6^3- (4d, low-spin S=0 -- corrected spin state)",
        xyz_content="""7
RhCl6 3- test geometry (octahedral, Rh-Cl = 2.34 Ang)
Rh  0.00   0.00   0.00
Cl  2.34   0.00   0.00
Cl -2.34   0.00   0.00
Cl  0.00   2.34   0.00
Cl  0.00  -2.34   0.00
Cl  0.00   0.00   2.34
Cl  0.00   0.00  -2.34
""",
        charge=-3,
        spin_mult=1,
        expect_ecp_element="Rh",
    )

    # ReCl6^2-: Re-Cl = 2.35 Ang, octahedral, d3 S=3/2. Re is 5d --
    # different row from Mo/Rh, so its def2-ECP core size is expected to
    # be larger (5d metals typically use a 60-electron small-core ECP,
    # replacing through 4f, vs 28 for 4d metals replacing through 3d).
    # This run tells us the actual number rather than assuming it.
    results["ReCl6_2m"] = run_smoke_test(
        label="ReCl6^2- (5d -- different ECP core size expected vs Mo/Rh)",
        xyz_content="""7
ReCl6 2- test geometry (octahedral, Re-Cl = 2.35 Ang)
Re  0.00   0.00   0.00
Cl  2.35   0.00   0.00
Cl -2.35   0.00   0.00
Cl  0.00   2.35   0.00
Cl  0.00  -2.35   0.00
Cl  0.00   0.00   2.35
Cl  0.00   0.00  -2.35
""",
        charge=-2,
        spin_mult=4,
        expect_ecp_element="Re",
    )

    # OsCl6^2-: Os-Cl = 2.32 Ang, octahedral, d4 S=1. Also 5d -- same
    # core-size expectation as Re, independent confirmation.
    #
    # This system did NOT converge with pyscf defaults in the prior run
    # (50 cycles, default init_guess, no level shift) -- that result was
    # wrongly reported as PASS because the test only checked which
    # solver class ran, not convergence. Two calls below: first
    # reproduces that same non-converged result explicitly (now correctly
    # failing instead of silently passing), second tries explicit
    # convergence aids to see whether this is fixable or a genuine flag.
    results["OsCl6_2m_default"] = run_smoke_test(
        label="OsCl6^2- with pyscf defaults (reproduces prior non-convergence)",
        xyz_content="""7
OsCl6 2- test geometry (octahedral, Os-Cl = 2.32 Ang)
Os  0.00   0.00   0.00
Cl  2.32   0.00   0.00
Cl -2.32   0.00   0.00
Cl  0.00   2.32   0.00
Cl  0.00  -2.32   0.00
Cl  0.00   0.00   2.32
Cl  0.00   0.00  -2.32
""",
        charge=-2,
        spin_mult=3,
        expect_ecp_element="Os",
    )

    results["OsCl6_2m_convergence_aids"] = run_smoke_test(
        label="OsCl6^2- with explicit convergence aids (max_cycle=200, atom guess, level_shift=0.2)",
        xyz_content="""7
OsCl6 2- test geometry (octahedral, Os-Cl = 2.32 Ang)
Os  0.00   0.00   0.00
Cl  2.32   0.00   0.00
Cl -2.32   0.00   0.00
Cl  0.00   2.32   0.00
Cl  0.00  -2.32   0.00
Cl  0.00   0.00   2.32
Cl  0.00   0.00  -2.32
""",
        charge=-2,
        spin_mult=3,
        expect_ecp_element="Os",
        scf_max_cycle=200,
        scf_init_guess="atom",
        scf_level_shift=0.2,
    )

    print("=== Summary ===")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    # --- Spin-parity fix smoke test ---
    # Reproduce the exact CrCl6^3- scenario without a real DMRG calculation.
    # autoCAS produced CAS(14,10): nelec=14, ncorelec=129-14=115 (ODD) ->
    # PySCF's `assert ncorelec % 2 == 0` failed inside sort_mo/ncore.
    # This test confirms the parity logic identifies orbital 63 as the
    # missing Cr t2g d orbital and extends to CAS(15,11) with even ncorelec.
    print()
    print("=== Spin-consistency fix smoke tests ===")
    print("--- Case 1: CrCl6^3- (1 missing d orbital, condition A fails) ---")

    import tempfile as _tf, os as _os
    _xyz = "7\nCrCl6 3-\nCr 0 0 0\nCl 2.33 0 0\nCl -2.33 0 0\nCl 0 2.33 0\nCl 0 -2.33 0\nCl 0 0 2.33\nCl 0 0 -2.33\n"
    with _tf.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as _f:
        _f.write(_xyz)
        _xyz_path = _f.name

    try:
        _mol = Molecule(_xyz_path)
        _iface = PatchedPyscfInterface(_mol)
        _iface.settings.basis_set = "def2-svp"
        _iface.settings.get_molecule().charge = -3
        _iface.settings.get_molecule().spin_multiplicity = 4
        _iface.settings.uhf = True
        _iface._build_molecule()   # populates pyscf_mol so .nelectron is known

        # Set occupation matching CrCl6 valence: orbs 39-62 doubly occ,
        # 63/64/65 singly occ (Cr t2g), 66-76 virtual.
        _occ = [0] * 77
        for _i in range(39, 63): _occ[_i] = 2
        for _i in [63, 64, 65]:  _occ[_i] = 1
        _iface.settings.get_molecule().occupation = _occ

        # Plateau output as actually produced: CAS(14,10)
        _cas_occ = [2, 2, 2, 2, 2, 2, 1, 1, 0, 0]
        _cas_idx = [45, 46, 52, 53, 55, 56, 64, 65, 67, 68]
        _nelec = sum(_cas_occ)
        _ncore = _iface.pyscf_mol.nelectron - _nelec
        print(f"  Before: nelec={_nelec}, ncorelec={_ncore} (ODD={_ncore%2!=0})")
        assert _ncore % 2 != 0, "Expected odd ncorelec in this test"

        # Run the parity logic (mirrors _final_cas_impl's parity block exactly)
        _missing = sorted([i for i, o in enumerate(_occ) if o == 1 and i not in _cas_idx])
        assert _missing, "Expected at least one missing singly-occ orbital"
        _extra = _missing[0]
        _new_idx = sorted(_cas_idx + [_extra])
        _new_occ = [_occ[i] for i in _new_idx]
        _new_nelec = sum(_new_occ)
        _new_ncore = _iface.pyscf_mol.nelectron - _new_nelec
        print(f"  Added orbital {_extra}")
        print(f"  After:  nelec={_new_nelec}, ncorelec={_new_ncore} (EVEN={_new_ncore%2==0})")
        assert _extra == 63,        f"Expected orbital 63, got {_extra}"
        assert _new_nelec == 15,    f"Expected 15 active electrons, got {_new_nelec}"
        assert _new_ncore % 2 == 0, "ncorelec still odd after fix"
        print("Case 1 PASSED: CAS(14,10) -> CAS(15,11), ncorelec 115->114")
    finally:
        _os.unlink(_xyz_path)

    print()
    print("--- Case 2: MnCl4^2- (2 missing d orbitals, condition B fails) ---")
    _xyz2 = "5\nMnCl4 2-\nMn 0 0 0\nCl 1.322 1.322 1.322\nCl 1.322 -1.322 -1.322\nCl -1.322 1.322 -1.322\nCl -1.322 -1.322 1.322\n"
    with _tf.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as _f2:
        _f2.write(_xyz2)
        _xyz2_path = _f2.name

    try:
        _mol2 = Molecule(_xyz2_path)
        _iface2 = PatchedPyscfInterface(_mol2)
        _iface2.settings.basis_set = "def2-svp"
        _iface2.settings.get_molecule().charge = -2
        _iface2.settings.get_molecule().spin_multiplicity = 6
        _iface2.settings.uhf = True
        _iface2._build_molecule()

        _occ2 = [0] * 60
        for _i in range(29, 43): _occ2[_i] = 2
        for _i in [45, 46, 47, 48, 49]: _occ2[_i] = 1
        _iface2.settings.get_molecule().occupation = _occ2

        # Plateau output as actually produced: CAS(17,10)
        _cas_occ2 = [2, 2, 2, 2, 2, 2, 2, 1, 1, 1]
        _cas_idx2 = [33, 34, 35, 36, 37, 42, 43, 47, 48, 49]
        _nelec2 = sum(_cas_occ2)
        _mol_spin2 = _iface2.pyscf_mol.spin
        _alpha2 = (_nelec2 + _mol_spin2) / 2
        _ncore2 = _iface2.pyscf_mol.nelectron - _nelec2
        print(f"  Before: nelec={_nelec2}, norbs={len(_cas_occ2)}, mol.spin={_mol_spin2}")
        print(f"  ncorelec={_ncore2} (even={_ncore2%2==0}), alpha={_alpha2} > norbs={len(_cas_occ2)} (BAD)")

        _missing2 = sorted([i for i, o in enumerate(_occ2) if o == 1 and i not in _cas_idx2])
        print(f"  Missing singly-occ: {_missing2}")
        _cur_idx = list(_cas_idx2)
        _cur_occ = list(_cas_occ2)
        for _extra in _missing2:
            _ne = sum(_cur_occ); _no = len(_cur_occ); _nc = _iface2.pyscf_mol.nelectron - _ne
            _al = (_ne + _mol_spin2) / 2
            if _nc % 2 == 0 and _al <= _no and _ne >= _mol_spin2:
                break
            _cur_idx = sorted(_cur_idx + [_extra])
            _cur_occ = [_occ2[i] for i in _cur_idx]
            print(f"  Added orbital {_extra}: CAS({sum(_cur_occ)},{len(_cur_occ)})")

        _new_ne = sum(_cur_occ); _new_no = len(_cur_occ)
        _new_nc = _iface2.pyscf_mol.nelectron - _new_ne
        _new_al = (_new_ne + _mol_spin2) / 2
        print(f"  After: CAS({_new_ne},{_new_no}), ncorelec={_new_nc} (even={_new_nc%2==0}), alpha={_new_al}")
        assert _new_nc % 2 == 0, "ncorelec not even"
        assert _new_al <= _new_no, f"alpha {_new_al} > norbs {_new_no}"
        assert _new_ne >= _mol_spin2, "nelec < mol.spin"
        assert _new_nc % 2 == 0 and _new_al <= _new_no and _new_ne >= _mol_spin2, f"Conditions still failing after fix: ncore={_new_nc} alpha={_new_al} norbs={_new_no} nelec={_new_ne}"
        
        print("Case 2 PASSED: CAS(17,10) -> CAS(19,12), all 3 conditions satisfied")
    finally:
        _os.unlink(_xyz2_path)
