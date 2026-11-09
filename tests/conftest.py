"""Shared test configuration."""

import os
import sys

from hypothesis import settings

# Importing code must never leave bytecode behind: modules imported from source/ would
# otherwise create __pycache__ directories there.
sys.dont_write_bytecode = True

# Fixed example sequences keep the commit-level run reproducible; the nightly profile explores more.
settings.register_profile("commit", max_examples=100, deadline=None, derandomize=True)
settings.register_profile("nightly", max_examples=2000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "commit"))
