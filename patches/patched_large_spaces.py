#!/usr/bin/env python3
"""
patched_large_spaces.py

Monkeypatch for a second confirmed bug in the installed scine_autocas
3.0.0, in:
  /pc2/users/h/hpcmual/.local/lib/python3.10/site-packages/scine_autocas/cas_selection/large_active_spaces.py
  LargeSpaces._partition_space (lines 64-99)

THE BUG (confirmed by direct, timeout-bounded reproduction -- see
verify_partition_bug.py output: the unpatched function hung past a 5s
timeout on CrCl6's exact numbers; the patch below returned instantly
on the identical input):

  When a sub-CAS subvector needs padding up to `int(max_orbitals / 2)`
  elements but the source orbital list is smaller than that target, the
  "random fill" fallback (original lines 88-97) retries
  `np.random.randint(...)` until it finds an index not already in the
  current subvector. If the subvector has ALREADY come to contain every
  element of orbital_indices (which happens whenever len(orbital_indices)
  < target and all of it got consumed in this same subvector), no random
  pick can ever succeed -- the retry loop has no escape condition and
  spins forever.

  This is exactly what happened to CrCl6^3-: 11 virtual orbitals, but
  the singly-occupied-orbital adjustment (max_orbitals 30 -> 27, then
  +1 for the odd-number correction) pushes the virtual-orbital target
  subvector size to 14. All 11 real virtual orbitals get consumed in
  the first 11 of 14 slots, then the remaining 3 slots try to randomly
  pad from a pool that is already 100% present in the subvector --
  genuine infinite loop, 100% CPU, zero further output, no dmrg_0
  directory ever created. This matches every symptom observed in job
  33397282 exactly.

THE FIX: before attempting a random pick, compute which orbitals (if
any) are NOT yet in the subvector. If none remain, stop padding and
accept a smaller-than-target subvector instead of retrying forever.
This is the smallest change that removes the infinite-loop failure
mode while preserving the original padding behavior whenever padding
is actually possible.

Usage: import and call apply_patch() BEFORE the autoCAS workflow runs
(same pattern as patched_pyscf_interface.py's monkeypatch of
PyscfInterface) -- this patches the LargeSpaces class itself, which
affects every instance, including ones already constructed by
run_action(), since Python method lookup happens at call time via the
class's __dict__.
"""

from typing import List

import numpy as np


def _patched_partition_space(self, orbital_indices: List[int]) -> List[List[int]]:
    """Create small active spaces within the valence space (patched).

    Same intent as the original LargeSpaces._partition_space: pad each
    subvector up to int(self.max_orbitals / 2) elements using random
    not-yet-used orbitals when the space doesn't divide evenly. Fixed
    to stop padding gracefully (accepting a smaller subvector) once
    every available orbital is already present, instead of spinning
    forever trying to find one that isn't.
    """
    partial_orbital_indices: List[List[int]] = []
    i = 0
    n = len(orbital_indices)
    target = int(self.max_orbitals / 2)

    while i < n:
        subvector: List[int] = []
        for _ in range(target):
            if i < n:
                subvector.append(orbital_indices[i])
                i += 1
            else:
                # Original intent: pad with a random orbital not yet in
                # this subvector. Fixed: if literally none remain
                # (subvector already contains every orbital in this
                # space), stop padding instead of looping forever.
                remaining = [x for x in orbital_indices if x not in subvector]
                if not remaining:
                    break
                if self.seed is not None:
                    np.random.seed(self.seed)
                pick = remaining[np.random.randint(len(remaining))]
                subvector.append(pick)
        partial_orbital_indices.append(subvector)

    return partial_orbital_indices


def apply_patch():
    """Monkeypatch LargeSpaces._partition_space at the class level.

    Must be called before the autoCAS workflow actually runs
    (get_large_active_spaces -> generate_spaces -> _partition_space),
    but can be called any time before that, regardless of whether a
    LargeSpaces instance already exists -- this patches the class
    itself, not an instance.
    """
    from scine_autocas.cas_selection.large_active_spaces import LargeSpaces

    original = LargeSpaces._partition_space
    LargeSpaces._partition_space = _patched_partition_space
    print(
        f"[patch] LargeSpaces._partition_space: {original} -> {_patched_partition_space}"
    )
    assert LargeSpaces._partition_space is _patched_partition_space, "Monkeypatch did not take"


if __name__ == "__main__":
    # Standalone re-verification: reproduce the exact CrCl6 failure case
    # against the REAL installed LargeSpaces class (not the isolated
    # copy used in verify_partition_bug.py), confirming the patch
    # works against the actual class we're patching, not just a
    # lookalike.
    import signal

    class TimeoutErr(Exception):
        pass

    def _handler(signum, frame):
        raise TimeoutErr()

    from scine_autocas.cas_selection.large_active_spaces import LargeSpaces

    ls = LargeSpaces()
    ls.max_orbitals = 28  # matches CrCl6's actual computed value
    ls.seed = 42
    virtual_orbitals = list(range(27, 38))  # 11 orbitals, matching CrCl6's actual count
    assert len(virtual_orbitals) == 11

    print("=== BEFORE patch: real LargeSpaces._partition_space on CrCl6's exact case ===")
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(5)
    try:
        result = ls._partition_space(virtual_orbitals)
        signal.alarm(0)
        print(f"UNEXPECTED: returned without hanging: {result}")
    except TimeoutErr:
        print("CONFIRMED on the real class: hung past 5s timeout.")

    print()
    print("=== Applying patch ===")
    apply_patch()

    print()
    print("=== AFTER patch: same call, same instance ===")
    result = ls._partition_space(virtual_orbitals)
    print(f"Result: {result}")
    print(f"Number of sub-spaces: {len(result)}")
    for idx, sub in enumerate(result):
        print(f"  sub-space {idx}: {len(sub)} orbitals -> {sub}")
