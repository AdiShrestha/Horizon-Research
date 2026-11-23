# Engineering-pilot-only files (IC-02 WP-A)

These files belong to the engineering phase. They are not used for Phase 1 production data and
are left byte-for-byte unchanged because frozen engineering evidence hashes them. This sidecar
note marks them instead of an in-file docstring.

- `src/trajcp/data/exponax_gen.py`: engineering pilot generator. Pilot seeds (20261124-20261137,
  tag 0x50325031) are engineering-only (A02). It is superseded for production by
  `src/trajcp/data/burgers.py` (ported from the parallel workspace, A03/A05).
- `configs/task/p2_burgers1d_pilot.json`: pilot prior (nu 5e-4 to 5e-3, 512-point solver, macro step
  0.005). It is not the A03 production prior (nu 1.75e-3 to 5e-3, 1,024-point solver, step 0.01).
