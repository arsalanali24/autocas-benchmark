#!/usr/bin/env python3
"""
analyze_crcl6_plateau.py  (v2 — adds diagonal check, second-plateau orbital
analysis, and generic per-system summary)

Extracts the mutual information matrix and s1 entropy values from any
autoCAS LargeCasWorkflow log and computes plateau widths numerically on
the MI axis. Closes the open loop from the CrCl6 review: the visual
"25 vs 27 step" estimate was not a measurement; this script provides one.

Also checks the second-qualifying plateau's orbital indices and occupations
so the reviewer can verify whether the second-plateau space is physically
plausible, the same way was done for the first plateau.

Run on Noctua2:
    source ~/.autocas_env.sh
    python3 analyze_crcl6_plateau.py \\
        /scratch/hpc-prf-qehpc/hpcmual/autocas_scratch/gold_crcl6_33447314.log \\
        --n-orbs 38 --valence-start 39 \\
        --cas-indices 45 46 52 53 55 56 64 65 67 68 \\
        --cas-occ    2  2  2  2  2  2  1  1  0  0  \\
        --parity-fixed-indices 45 46 52 53 55 56 63 64 65 67 68 \\
        --parity-fixed-occ     2  2  2  2  2  2  1  1  1  0  0  \\
        --mol-spin 3

For a new system (e.g. MoCl6 after it completes):
    python3 analyze_crcl6_plateau.py \\
        /scratch/.../gold_mocl6_33447319.log \\
        --n-orbs 38 --valence-start 54 \\
        --cas-indices 59 60 65 66 68 69 72 73 74 75 76 77 78 79 \\
        --cas-occ    2  2  2  2  2  2  1  1  1  0  0  0  0  0  \\
        --mol-spin 3
"""

import sys
import re
import argparse
import numpy as np


def extract_matrix(log_text, keyword, n):
    """
    Extract an n×n numpy array printed by numpy after `keyword`.
    Handles truncated (...) output by warning when fewer than n*n values
    are found.
    """
    start = log_text.find(keyword)
    if start == -1:
        return None, "keyword not found"
    block = log_text[start + len(keyword):]
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', block[:n * n * 30])
    if len(nums) < n * n:
        return None, (f"only {len(nums)} values found, need {n*n}. "
                      f"Matrix was likely truncated by numpy (set np.set_printoptions"
                      f"(threshold=np.inf) in the producing code, or extract from .h5).")
    mat = np.array([float(x) for x in nums[:n * n]]).reshape(n, n)
    return mat, "ok"


def extract_s1(log_text, n):
    keyword = "initial s1:"
    start = log_text.find(keyword)
    if start == -1:
        return None, "keyword not found"
    block = log_text[start + len(keyword):]
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', block[:n * 30])
    if len(nums) < n:
        return None, f"only {len(nums)} values found, need {n}"
    return np.array([float(x) for x in nums[:n]]), "ok"


def plateau_width_report(values, label, threshold_step=0.01, plateau_min=10,
                         valence_start=0, mol_occ=None, mol_spin=0):
    """
    Scan threshold from 0 to max(values)+step in steps of threshold_step.
    A plateau is a run of ≥ plateau_min consecutive steps with the same
    orbital count. Returns list of plateau dicts.

    mol_occ: dict {orbital_index: occupation} for cross-checking plateaus.
    mol_spin: 2S value for spin-consistency check on each plateau.
    valence_start: offset to convert internal index to orbital number.
    """
    tmax = float(np.ceil(values.max() / threshold_step) * threshold_step) + threshold_step
    thresholds = np.arange(0, tmax, threshold_step)
    # (threshold, n_selected, set_of_internal_indices_selected)
    curve = []
    for t in thresholds:
        sel = np.where(values > t)[0]
        curve.append((float(t), len(sel), frozenset(sel.tolist())))

    print(f"\n{'='*65}")
    print(f"Criterion: {label}")
    print(f"{'='*65}")
    print(f"Range: [{values.min():.4f}, {values.max():.4f}]")
    print(f"plateau_min = {plateau_min} steps   threshold_step = {threshold_step}")

    # Print the staircase (only at transitions)
    print(f"\n{'Threshold':>10}  {'N selected':>10}  {'Change':>8}")
    prev_n = None
    for t, n, _ in curve:
        if n != prev_n:
            change = f"→ {n}" if prev_n is not None else "start"
            print(f"{t:10.3f}  {n:10d}  {change:>8}")
            prev_n = n

    # Find plateaus
    plateaus = []
    i = 0
    while i < len(curve):
        t0, n0, idx0 = curve[i]
        j = i + 1
        while j < len(curve) and curve[j][1] == n0:
            j += 1
        width = j - i
        # The set of selected orbitals is stable across this run
        plateau_entry = {
            'n_orbitals': n0,
            'n_steps': width,
            't_start': curve[i][0],
            't_end': curve[j-1][0],
            'orbital_internal_indices': sorted(idx0),
            'orbital_numbers': sorted([vi + valence_start for vi in idx0]),
        }
        if mol_occ is not None:
            occ_list = [mol_occ.get(orb, 'unknown')
                        for orb in plateau_entry['orbital_numbers']]
            n_singly = sum(1 for o in occ_list if o == 1)
            n_doubly = sum(1 for o in occ_list if o == 2)
            n_virt   = sum(1 for o in occ_list if o == 0)
            nelec = n_doubly * 2 + n_singly
            mol_ne = sum(mol_occ.values())  # rough total from CAS only
            plateau_entry['occ_summary'] = (n_doubly, n_singly, n_virt)
            plateau_entry['nelec_in_cas'] = nelec
            plateau_entry['n_singly'] = n_singly
            plateau_entry['spin_consistent'] = (n_singly >= mol_spin)
        if width >= plateau_min:
            plateau_entry['qualifies'] = True
            plateaus.append(plateau_entry)
        else:
            plateau_entry['qualifies'] = False
        i = j

    # Print qualifying plateaus
    print(f"\nQUALIFYING PLATEAUS (>= {plateau_min} steps):")
    if not plateaus:
        print("  none found")
        return []

    for rank, p in enumerate(plateaus, 1):
        marker = " <-- FIRST (autoCAS selects this)" if rank == 1 else ""
        print(f"\n  Plateau {rank}{marker}")
        print(f"    Orbitals selected: {p['n_orbitals']}")
        print(f"    Width:             {p['n_steps']} steps "
              f"(T={p['t_start']:.3f} to T={p['t_end']:.3f})")
        print(f"    Orbital numbers:   {p['orbital_numbers']}")
        if 'occ_summary' in p:
            nd, ns, nv = p['occ_summary']
            print(f"    Occupation:        {nd} doubly + {ns} singly + {nv} virtual")
            print(f"    Active electrons:  {p['nelec_in_cas']}")
            sc = "OK" if p['spin_consistent'] else f"FAIL (need >= {mol_spin} singly occ)"
            print(f"    Spin-consistent:   {sc}  (mol.spin={mol_spin})")
        if rank == 2:
            print(f"    *** Second plateau — the 'alternative' discussed in review ***")
    return plateaus


def diagonal_check(mi, label="MI matrix"):
    """
    Report diagonal values. For autoCAS MI matrices, I_ii is typically
    s1[i] (single-orbital entropy) or 0. Either way, the 'largest element'
    criterion should use max_{j != i}(I_ij), not max_j(I_ij), to avoid
    the diagonal inflating the criterion for each orbital.

    This function zeros the diagonal (in-place) and reports what was there.
    """
    diag = np.diag(mi).copy()
    print(f"\n{'='*65}")
    print(f"Diagonal check — {label}")
    print(f"{'='*65}")
    print(f"Diagonal range: [{diag.min():.4f}, {diag.max():.4f}]")
    if diag.max() > 1e-10:
        print(f"Non-zero diagonal detected. Setting to zero before computing "
              f"max_j(I_ij) per orbital (excluding self-correlation).")
        print(f"Diagonal values: {np.round(diag, 4)}")
    else:
        print(f"Diagonal is effectively zero — no impact on max_j calculation.")
    np.fill_diagonal(mi, 0.0)
    return diag


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log', help='Path to autoCAS job log file')
    parser.add_argument('--n-orbs', type=int, required=True,
                        help='Number of orbitals in the full valence CAS window')
    parser.add_argument('--valence-start', type=int, default=0,
                        help='0-based index of first valence orbital (e.g. 39 for CrCl6)')
    parser.add_argument('--cas-indices', type=int, nargs='+', default=[],
                        help='autoCAS-selected orbital indices (0-based, before parity fix)')
    parser.add_argument('--cas-occ', type=int, nargs='+', default=[],
                        help='autoCAS-selected occupations (matching --cas-indices)')
    parser.add_argument('--parity-fixed-indices', type=int, nargs='+', default=[],
                        help='Orbital indices after spin-consistency fix (if applied)')
    parser.add_argument('--parity-fixed-occ', type=int, nargs='+', default=[],
                        help='Occupations after spin-consistency fix')
    parser.add_argument('--mol-spin', type=int, default=0,
                        help='mol.spin = 2S (number of unpaired electrons)')
    parser.add_argument('--plateau-min', type=int, default=10,
                        help='Minimum plateau width in steps (plateau_values in YAML)')
    parser.add_argument('--threshold-step', type=float, default=0.01,
                        help='Threshold scan step (threshold_step in YAML)')
    args = parser.parse_args()

    print(f"Log: {args.log}")
    with open(args.log) as f:
        log_text = f.read()

    N = args.n_orbs
    vs = args.valence_start

    # Build occupation dict from cas-indices/occ (pre-fix, for cross-checking)
    mol_occ = {}
    if args.cas_indices and args.cas_occ:
        assert len(args.cas_indices) == len(args.cas_occ), \
            "--cas-indices and --cas-occ must have same length"
        mol_occ = dict(zip(args.cas_indices, args.cas_occ))

    # Extract s1
    s1, s1_status = extract_s1(log_text, N)
    if s1 is None:
        print(f"ERROR extracting s1: {s1_status}")
        sys.exit(1)
    print(f"\ns1 extracted: {N} values, max={s1.max():.4f}, min={s1[s1>0].min():.4f}")
    print(f"Max s1 = {s1.max():.4f} at internal index {s1.argmax()} "
          f"(orbital {s1.argmax() + vs})")
    print(f"NOTE: if the threshold plot x-axis extends past {s1.max():.2f}, "
          f"the x-axis cannot be raw s1 — confirmed independent of any width argument.")

    # Extract MI matrix
    mi, mi_status = extract_matrix(log_text, "initial_mutual_information:", N)
    if mi is None:
        print(f"\nERROR extracting MI matrix: {mi_status}")
        print("Cannot compute MI-axis plateau widths without the full matrix.")
        print("Fallback: computing s1-axis only.")
        print("\nTo get the full matrix, either:")
        print("  1. Add np.set_printoptions(threshold=np.inf) before the print in the workflow")
        print("  2. Extract directly from the QCMaquis result h5 file:")
        print("     python3 -c \"import h5py,numpy as np; f=h5py.File('qcmaquis_result_file.h5')")
        print("     print(list(f.keys()))\"")
        # Fall back to s1 only
        plateau_width_report(s1, "s1 (fallback — MI not available)",
                             args.threshold_step, args.plateau_min,
                             vs, mol_occ, args.mol_spin)
        return

    # Diagonal check and zeroing
    diag_vals = diagonal_check(mi, "initial_mutual_information")

    # Compute max_{j != i}(I_ij) per orbital (diagonal already zeroed)
    mi_max = mi.max(axis=1)

    # Run plateau analysis on MI axis
    print(f"\n{'#'*65}")
    print("PLATEAU ANALYSIS — MI AXIS (the real x-axis of the threshold plot)")
    print(f"{'#'*65}")
    mi_plateaus = plateau_width_report(
        mi_max, "max_j(I_ij) per orbital (diagonal excluded)",
        args.threshold_step, args.plateau_min, vs, mol_occ, args.mol_spin)

    # Run plateau analysis on s1 for comparison
    print(f"\n{'#'*65}")
    print("PLATEAU ANALYSIS — s1 AXIS (comparison only; not the plot x-axis)")
    print(f"{'#'*65}")
    s1_plateaus = plateau_width_report(
        s1, "single-orbital entropy s1",
        args.threshold_step, args.plateau_min, vs, mol_occ, args.mol_spin)

    # Summary
    print(f"\n{'='*65}")
    print("FINAL SUMMARY")
    print(f"{'='*65}")

    if mi_plateaus:
        first = mi_plateaus[0]
        widest = max(mi_plateaus, key=lambda p: p['n_steps'])
        print(f"\nMI axis:")
        print(f"  First qualifying plateau:  {first['n_orbitals']} orbitals, "
              f"{first['n_steps']} steps")
        print(f"  Widest qualifying plateau: {widest['n_orbitals']} orbitals, "
              f"{widest['n_steps']} steps")
        if first['n_orbitals'] == widest['n_orbitals']:
            print(f"  VERDICT: first == widest ({first['n_orbitals']} orbs). "
                  f"Selection is unambiguous on MI axis.")
        else:
            print(f"  VERDICT: first ({first['n_orbitals']}) != widest ({widest['n_orbitals']}). "
                  f"autoCAS selects first; the widest would give a different space.")
            if len(mi_plateaus) > 1:
                second = mi_plateaus[1]
                print(f"  Second plateau ({second['n_orbitals']} orbs, "
                      f"{second['n_steps']} steps) spin-consistent: "
                      f"{second.get('spin_consistent', 'unknown')}")

    if s1_plateaus:
        first_s1 = s1_plateaus[0]
        print(f"\ns1 axis (for comparison):")
        print(f"  First qualifying plateau: {first_s1['n_orbitals']} orbs, "
              f"{first_s1['n_steps']} steps")
        if mi_plateaus:
            match = "MATCH" if first_s1['n_orbitals'] == mi_plateaus[0]['n_orbitals'] \
                    else "MISMATCH (expected if s1 != MI criterion)"
            print(f"  vs MI first: {match}")

    if args.parity_fixed_indices and args.parity_fixed_occ:
        print(f"\nSpin-consistency fix applied:")
        pre_n  = len(args.cas_indices)
        post_n = len(args.parity_fixed_indices)
        pre_e  = sum(args.cas_occ)
        post_e = sum(args.parity_fixed_occ)
        print(f"  Before fix: CAS({pre_e}, {pre_n})")
        print(f"  After fix:  CAS({post_e}, {post_n})")
        added = set(args.parity_fixed_indices) - set(args.cas_indices)
        print(f"  Added orbitals: {sorted(added)}")
        post_singly = sum(1 for o in args.parity_fixed_occ if o == 1)
        print(f"  Singly occ after fix: {post_singly} (mol.spin={args.mol_spin} required)")
        print(f"  Spin-consistent: {'YES' if post_singly >= args.mol_spin else 'NO'}")


if __name__ == "__main__":
    main()
