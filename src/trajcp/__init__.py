"""trajcp: trajectory-level conformal prediction bands for autoregressive neural PDE emulators.

Layout:

* ``trajcp.data``        reference-solution generation, priors and storage
* ``trajcp.emulators``   neural emulators and their training
* ``trajcp.rollout``     batched rollouts, sensitivity fields and per-trajectory statistics
* ``trajcp.conformal``   calibration core (float64, CPU, NumPy only)
* ``trajcp.evaluation``  metrics and statistical procedures
* ``trajcp.plotting``    figure styles and helpers
"""

__version__ = "0.1.0.dev0"
