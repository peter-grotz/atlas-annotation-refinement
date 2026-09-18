"""Every public entry point must import first, in a fresh interpreter.

`atlas_refine.algorithms` and `atlas_refine.evaluate` import from one another,
and Python tolerates that only if one side finishes loading before the other
is entered. Whichever module happens to be imported first in a given process
decides which side that is - so a cycle here does not show up by importing
both in the same test process, only by being the *first* thing imported in a
fresh one. pytest's own collection order imports ``atlas_refine.algorithms``
before ``atlas_refine.evaluate`` on every run, which is exactly why this broke
silently: the suite could pass indefinitely while `from atlas_refine.evaluate
import fit` as a script's first line raised ImportError.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

ENTRY_POINTS = [
    "import atlas_refine",
    "from atlas_refine.algorithms import REGISTRY",
    "from atlas_refine.evaluate import fit",
    "from atlas_refine.characterize import characterize",
    "from atlas_refine.review import screen_label",
    "from atlas_refine.registration import register",
    "from atlas_refine.analysis import profile_image",
    "from atlas_refine.io import GroundTruthStore",
]


@pytest.mark.parametrize("statement", ENTRY_POINTS)
def test_imports_first_in_a_fresh_interpreter(statement):
    result = subprocess.run(
        [sys.executable, "-c", statement], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
