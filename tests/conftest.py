"""Shared test configuration."""

import sys

# Importing code must never leave bytecode behind: modules imported from source/ would
# otherwise create __pycache__ directories there.
sys.dont_write_bytecode = True
