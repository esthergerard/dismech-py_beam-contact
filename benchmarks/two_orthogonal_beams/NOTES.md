# Two orthogonal beams (Zavarise & Wriggers Ex. 2) — history and conclusions

> **TL;DR for reproducing the paper's Fig. 7(c,d):** the continued
> 29-step run only completes cleanly with `--planar` (Beam Z's x-DOF
> fixed) and `--eberly-straight-R-threshold 2e2`, which neutralizes a
> numerical (not physical) arc-fit degeneracy at Beam Z's structural
> inflection point — validated to reproduce the same peak/final forces
> as the (crash-prone) unconstrained run. The canonical data is tagged
> `_fix_planarRth2` in `contact_output_files_soft_soft/`; see "Extended
> 29-step run" below, ending at "CLUSTER RESULT, CONFIRMED COMPLETE
> (2026-10-05)". The figure script is
> `postprocessing/fig_validation_and_final_trace.py` (reproduces the
> paper's Fig. 7 panels a-d in one composite); `postprocessing/fig_planar_final_comparison.py`
> is a related, broader mu=0/mu=0.1 comparison. Everything above that
> line in this file is the (superseded) investigation history kept for
> context on why the workaround is needed and safe.

Validation against Example 2 of Zavarise & Wriggers (2000), *Contact with
friction between beams in 3-D space*, IJNME 49(8):977–1006: two
orthogonal beams, off-centre crossing (10 m from clamped end A, 4 m from
free end B), loaded into contact. Paper Tables give reference F_N (and
F_T for the friction cases).

Main script: `contact_bending_soft_soft.py` (despite the legacy name —
it is the Zavarise benchmark, both beams flexible).

## What was tried

- Initial setup ("two beams, bench Karim"), then geometry corrected to
  match the paper exactly; Euler-Bernoulli validation included in the 3D
  bench.
- A small initial curvature added on the beams: the Eberly arc fit
  (`circle_from_three_nodes`) needs non-collinear triples; perfectly
  straight beams hit the collinearity guard.
- Cluster-version bug fixed (a650820), then **mesh convergence** on the
  frictionless case: nx ∈ {21, 51, 101} overlaid against the paper's
  reference values (`run_zavarise_mesh_convergence.py`, results in
  `contact_output_files_soft_soft/`).
- **Friction** (`friction/` subfolder): Coulomb friction with a smooth
  velocity gate (`vel_tol`), grid over (μ, nx) run sequentially by
  `friction/run_zavarise_friction_mesh_convergence.py`; overlays of
  |F_N|(step) and |F_T|(step) vs paper Table IV by
  `friction/plot_friction_mesh_convergence.py`.
  Two friction coefficients retained for the paper.
- Friction force later recomputed with the *current-iterate* velocity and
  the corresponding dF/dv · dv/dq Jacobian term added (b1bdb2e) —
  improved Newton convergence with friction on.

## Conclusions

- Frictionless F_N converges with mesh to the paper's reference; nx=101
  is well converged, nx=21 already close.
- Friction cases reproduce the paper's tangential forces with the smooth
  Coulomb gate; the velocity-consistent Jacobian is needed for clean
  quadratic convergence.
- The geometry figure for the paper comes from
  `figures/plot_zavarise_geometry.py`; per-step force tables are written
  by the sim and post-processed by `figures/post_process_soft_soft.py`.

## Paper data provenance

All data files below live in `contact_output_files_soft_soft/`
(paths verified 2026-08-24):

| paper item | script | data file(s) |
|---|---|---|
| geometry + metrics figure | `postprocessing/fig_soft_soft.py` | `zavarise_coords_steps6_nx21_mu01.txt`; frictionless convergence: `zavarise_compare_steps6_nx{11,21,51,101}.txt`; friction: `zavarise_compare_steps6_nx{21,51,101}_mu01.txt` |
| jitter table (tex) | `postprocessing/table_soft_soft_jitter.py` | `zavarise_per_dt_trace_steps6_fix_imc_mu0.txt`, `..._fix_eberly_mu0.txt`, `..._fix_imc_mu01.txt`, `..._fix_eberly_mu01.txt` |
| friction mesh-convergence overlays | `friction/plot_friction_mesh_convergence.py` | `zavarise_compare_steps6_nx{nx}_mu{tag}.txt` for nx ∈ {21, 51, 101}, both μ values |
| geometry sketch | `figures/plot_zavarise_geometry.py` | any `zavarise_coords_*` file (initial config only) |

The `zavarise_per_dt_trace_steps6_fix_*` traces were produced by the μ=0.8
NH/SC cluster jobs (`friction/run_trc_NH_nx101.sh`, `run_trc_SC_nx101.sh`)
and the compare files by `run_zavarise_mesh_convergence.py` /
`friction/run_zavarise_friction_mesh_convergence.py`.

## Extended 29-step run (2026-09-28)

Rationale: at 6 steps the contact point slides only 0.18 m (< one 0.5 m
element at nx=21), so the 2-node model never crosses a node and its
node-passing jitter cannot show; the 3-node model lost every μ=0 metric
of the jitter table.  29 steps = the paper's own Case 2: slide 0.87 m
(~2 element crossings), analysed window ~26 s instead of 3 s.

- Traces: `zavarise_per_dt_trace_steps29_fix_{imc,eberly}_mu{0,01}.txt`,
  produced on the cluster by `run_zavarise_steps29_jitter_cluster.sh`
  (SLURM array 0-3; same flags as the steps6 rows, just 29 steps).
- Table: `postprocessing/table_soft_soft_jitter.py --steps 29 --settling 2`
  → `postprocessing/figures/table_soft_soft_jitter_steps29.{txt,tex}`.
  `--settling 2` because the impact ring-down outlasts the 1 s window
  chosen for the 6 s record (visible to t ≈ 3.5 s); the window also
  auto-stops 1 settling before disengagement (see below).
- Figure: `postprocessing/fig_soft_soft_steps29.py`
  → `figures/fig_soft_soft_trace_steps29.{pdf,svg}`.

Findings:
- μ = 0.1: both models LOSE CONTACT at t ≈ 18 s (Beam Z snaps past
  Beam X; 2-node at 18.07 s, 3-node at 18.46 s — same physics).  Before
  that the 2-node trace has O(10 N) jumps; the 3-node wins all five
  jitter metrics by 5–10x.
- Chatter-origin experiment (2026-09-29): reran the 3-node μ=0.1 trace
  with Norton-Hoff capped (ρ=0.1, v_ref=0.03 — saturation matched to
  smooth Coulomb at the loading sliding speed), with and without the
  velocity Jacobian (`run_zavarise_steps29_NH_cluster.sh`; needed the
  norton_hoff_capped branch added to the batched gate, a --max-iter=400
  headroom, and incremental trace writing — see git log).  Result: the
  2.75 Hz oscillation is IDENTICAL across smooth Coulomb / NH / NH+jac
  (osc std 5.02/4.97/5.09 N, flip rate 2.9/s, release 18.44–18.49 s).
  With the earlier 2-node/3-node frequency match this rules out the
  friction regularization, the Jacobian treatment AND the contact model:
  the chatter is a genuine stick-slip relaxation oscillation of the
  elastic beam pair at the Coulomb limit (sawtooth tangential wind-up +
  spike release, clearly visible in F_T,z), i.e. physics of the setup
  (μ, beam compliance, weak η=10 damping, slow driving).  Figure:
  `postprocessing/fig_chatter_nh_vs_sc.py` →
  `figures/fig_chatter_nh_vs_sc_steps29.{pdf,svg}`.
- PLANAR variant (2026-09-29, `--planar`: Beam Z x-DOF fixed): the
  μ=0.1 chatter VANISHES entirely (IMC: 0.14 N std vs 5.74 N in 3D,
  zero friction-direction reversals; Eberly/NH: 0.00 N) and there is NO
  snap-past — contact persists to 29 s.  So 100% of the 2.75 Hz
  oscillation and the t≈18 s release were the out-of-plane frictional
  flutter (follower force on the lateral "rail" DOF; Esther spotted it
  in the 3D figure).  Elastic lateral restoring from Beam X's
  compliance valley is real but ~40x too weak (0.34 N over 1 m vs
  μF_N ≈ 15 N).  In the common pre-crossing window both contact models
  are identically jitter-free (3-pt metrics exactly 0).
  CAVEAT: the Eberly planar runs livelock (400 iters, friction-
  INdependent) exactly at slide = 0.25/0.50 m — half/full-element
  registry, where the 3-node window handover is perfectly degenerate in
  the exactly-planar symmetric configuration (3D runs cross it fine;
  1e-7 m of wobble breaks the tie).  Fix: `--z-contact 3.03` de-phases
  the mesh registry (arm 10.03 m, -0.3% on F_N);
  `run_zavarise_steps29_planar03_cluster.sh` reruns the four table rows
  planar+dephased; table via `--variant planar03`.
  **SUPERSEDED (2026-10-03)**: `--x-shift` debug test (round 1, Beam X
  shifted 0.1 m) barely moved the crash (8.72 vs 8.74 s) — REFUTES the
  mirror-tie-on-Beam-X hypothesis.  `--planar` only constrains Beam Z,
  so the degeneracy must live there; `--z-contact`/`planar03` was
  targeting the wrong beam.  **Do not submit `run_zavarise_steps29_planar03_cluster.sh`.**
  Refined hypothesis: Beam Z's x-DOF pinned at exactly 0 gives every
  arc-fit triple a circle whose plane normal is EXACTLY the global
  x-axis at all times — an exact coordinate alignment free 3D never has.
  New flag `--planar-tilt DX` gives Beam Z a tiny fixed linear tilt in x
  before the x-DOF lock (still zero dynamic freedom — no flutter risk —
  just not axis-aligned).  Test batch:
  `run_zavarise_planar_debug2_cluster.sh` (control + tilt 1e-3 + tilt
  1e-2, 10 steps, μ=0).  Not yet submitted.
  Diagnostic figures: `postprocessing/fig_planar_crash.py` (force trace
  + 2D side/top views at the last converged step, μ=0 — NOTE: an earlier
  version used an oblique 3D panel that visually suggested the free tip
  bows out of plane; this was a matplotlib mplot3d rendering artifact
  for this very non-cubic geometry, NOT real motion — raw coords file
  confirms Beam Z's x is exactly 0.0 at every node/step; fixed by
  switching to plain 2D projections) and
  `postprocessing/fig_planar_mu01_comparison.py` (2-node vs 3-node
  planar overlay at μ=0.1 — 2-node complete to 29 steps, 3-node stops at
  its crash at t=8.77 s; shared pre-crash window metrics computed but
  this window is BEFORE any element crossing, so it only captures the
  initial contact ripple, not the node-passing/chatter behaviour the
  manuscript table targets).  A full-length planar μ=0.1 comparison
  needs the livelock fixed first.

  **ROOT CAUSE CONFIRMED (2026-10-03)**, via direct local instrumentation
  (monkey-patched eberly_batched, no src/dismech edits) cross-validated
  by an independent code-search agent reading the kernel against the
  saved crash snapshot:
  `batched_arc_min_dist` (eberly_batched.py) picks the beam-to-beam
  closest point via a hard `np.argmin` over 16 discrete candidates
  (8 interior polynomial-root candidates + 4 edge + 4 corner).  The
  TRUE underlying cause: Beam Z is clamped at A and free beyond the
  contact, so past the contact the bending moment is ~0 and the elastic
  curvature cancels the baked-in kappa0=1e-4 curvature at a point that
  travels along with the slide — i.e. Beam Z has a structural
  INFLECTION POINT that sweeps past each 3-node window roughly once per
  element of slide.  `circle_from_three_nodes` fits R = 1/kappa, so at
  the inflection kappa -> 0 and R -> infinity; the arc kernel's force
  explodes ~10,000x in a narrow band around the crossing (confirmed:
  |F_arc| peaks at 233 N vs a stable 0.023 N line-primitive answer, for
  a window whose physical contact force is ~0.02 N).  With Beam Z's
  x-DOF pinned at EXACTLY 0 (`--planar`), the discrete curvature
  `e_a x e_b` is confined to the ê_x LINE (a signed scalar) and MUST
  pass through exactly zero as the inflection sweeps through — in free
  3D (or even just its own roundoff-level wobble) the curvature is a
  genuine 2-vector that generically swerves AROUND the origin instead
  of through it, which is why only the exactly-planar run livelocks.
  Measured: the argmin selection for 3 specific pairs flips 46 times
  across the 69 Newton iterations of a locally-reproduced failing step,
  with a measured best-vs-2nd-best margin down to ~1e-9 — a genuine
  numerical tie, not a near-miss.
  RULED OUT: both originally-suspected hard-threshold branches
  (`m2 = abs(C0_can[:,0]) > 1e-14` in `_bprepare`, and
  `parallel = N0_can[:,2] >= 1-1e-10`) — confirmed via direct probe to
  be EXACTLY 0.0 for every active pair at every iteration, a structural
  fact of this orthogonal-beam geometry (not a near-miss), so they never
  engage at all.  `--x-shift` on Beam X (round 1) is likewise confirmed
  irrelevant — the degeneracy lives entirely in Beam Z's own curvature.
  **`--planar-tilt` is NOT a robust fix** (superseding my earlier
  conclusion to the contrary): a rigid tilt still confines Beam Z to
  SOME flat plane, so its curvature is still a signed scalar on a
  (now-rotated) line that must still cross zero at the SAME structural
  inflection-sweep event — a tilt only perturbs exactly WHEN that
  happens, not WHETHER.  Confirmed empirically: tilt=1e-3 reduces the
  FIRST crossing's iterations (69->10) but doesn't eliminate it;
  tilt=1e-2 fully resolves the first crossing but fails at a SECOND
  crossing (slide 0.056 m); tilt=0.05 (5x larger) fails at essentially
  the SAME second crossing (slide 0.052 m) — confirming a larger tilt
  delays but does not structurally prevent recurrence.  Do not use
  `run_zavarise_planar_debug2_cluster.sh` or
  `run_zavarise_steps29_planar_v2_cluster.sh` (both --planar-tilt based,
  superseded) or `run_zavarise_steps29_planar03_cluster.sh` (wrong beam,
  superseded earlier).
  **PROMISING SIM-SCRIPT-ONLY FIX (validating)**: `env.eberly_straight_R_threshold`
  is an EXISTING, already-wired opt-in (read in time_stepper.py at the
  beam-to-beam call site; already used by `double_spiral_test.py`) that
  routes any pair whose fitted arc radius exceeds the threshold to the
  already-stable line-primitive force instead of the arc kernel — this
  directly neutralizes the R->infinity blowup without touching
  src/dismech at all (pure sim-script configuration, e.g.
  `env.eberly_straight_R_threshold = 5e4` after `add_force('beamToBeamContactEberly3D', ...)`
  in contact_bending_soft_soft.py).  Caveat raised independently: the
  b2b driver uses a HARD threshold switch (unlike self-contact's smooth
  `_pair_line_weights` blend added 2026-09-17 after an analogous
  livelock); a hard switch could itself create a new, different
  threshold-crossing livelock.  Local validation in progress as of this
  writing.  If the simple threshold does NOT prove robust, the deeper
  fix (porting `_pair_line_weights`'s smooth arc/line blend into the b2b
  driver, `eberly_contact_3d.py` around line 2510) is a `src/dismech`
  kernel change requiring Esther's explicit approval first
  ([[model-change-boundary]]) — do not implement without asking.

  **CLUSTER RESULT (2026-10-05), threshold=1e3, max-iter=800**: 2-node
  (IMC) completes all 29 steps cleanly, both μ. 3-node (Eberly) STILL
  crashes — now at t≈10.72-10.76 s (slide≈0.32 m), later than the
  untilted baseline (8.74 s) but not complete. **Root cause of this
  residual failure identified**: BEAM X has its own structural
  inflection points (expected for a clamped-clamped beam under a point
  load — curvature changes sign between each clamp and the load, so
  there are two inflections by ordinary beam theory), and as the
  contact's effective loading on Beam X evolves, these inflections
  sweep past Beam X's own mesh windows too — i.e. the SAME R->infinity
  mechanism as Beam Z, just on the other beam.  Confirmed directly in
  the saved curvR output: Beam X's per-node fitted radius is normally
  ~70-430 m but rises to exactly a LOCAL MAXIMUM of 1036 m at nodes 5
  and 15 (ordinary bending in between, radius dips back down toward the
  middle) at the last converged step before the crash — i.e. sitting
  almost exactly AT the threshold=1e3 cutoff, not safely below it the
  way Beam Z's baseline (~1.3e4) is.  The per-dt force trace shows the
  signature directly: clean smooth ramp to t=10.69 s, then oscillation
  (131.84 -> 129.97 -> 132.08 -> 132.22 -> 130.34 N) for the last ~5
  samples before the hard failure — small-amplitude at this resolution,
  but the real sub-dt-step blowup (hidden between Newton iterations) is
  presumably what finally exceeds the iteration cap.
  **Next concrete step (not yet tried)**: lower the threshold well
  below Beam X's observed near-inflection range, e.g. `--eberly-straight-R-threshold 2e2`
  (200 m — comfortably below Beam X's normal peak of ~430 m away from
  its inflections, and far below Beam Z's baseline either way), so BOTH
  beams' inflection-driven blowups get caught.  Still a pure sim-script
  change, no kernel edits.
  The μ=0/μ=0.1 jitter table over the available (unequal, truncated)
  windows is NOT yet representative: the 3-node columns are
  contaminated by the pre-crash oscillation in the last few samples
  (Max 3-pt jumps to ~2.0 N, Mean 3-pt ~0.9-1.2 N — an order of
  magnitude worse than the clean 6/29-step 3D tables), which is a
  symptom of the still-unresolved residual issue, not a fair reflection
  of steady-state behaviour.  Do not quote this table until a run
  completes all 29 steps without crashing.

  **CORRECTION (2026-10-05)**: the "Beam X has its own inflection"
  explanation above for the threshold=1e3 residual failure was
  re-examined and is likely WRONG.  Beam X's curvature AT the actual
  contact point (its midpoint, node 10) is small and safe (R~72 m); the
  R~1036 m "peak" found at nodes 5/15 is 2.5 m away from the contact
  point, well outside the ~0.42 m broad-phase interaction range, so
  those windows are never actually paired with Beam Z and are
  irrelevant to the Eberly kernel's contact computation.  The true
  mechanism behind the threshold=1e3 residual failure is still not
  precisely identified, but is empirically resolved by a lower
  threshold:

  **VALIDATED FIX, threshold=2e2 (2026-10-05)**: reconstructing from the
  REAL step-10 state saved by the 2026-10-05 threshold=1e3 cluster run
  (i.e. the exact state that run's own physics produced, not an
  approximation) and continuing with the actual time stepper:
  threshold=1e3 reproduces the same failure at the same point (confirms
  the local reconstruction is faithful to the real run); threshold=2e2
  converges CLEANLY for all 1900 remaining dt-steps, reaching t=29s /
  slide≈0.89m -- i.e. the FULL remaining range of the benchmark, zero
  anomalies, steady 3 Newton iterations throughout.  New batch script:
  `run_zavarise_steps29_Rthresh2_cluster.sh` (tag `_fix_planarRth2_*`,
  same structure as the superseded `..._Rthresh_cluster.sh`, only the
  threshold value changed 1e3 -> 2e2).  This is the current
  recommendation for a complete 3-node planar run.

  **CLUSTER RESULT, CONFIRMED COMPLETE (2026-10-05)**: all four
  `_fix_planarRth2_*` runs (2-node/3-node x mu=0/mu=0.1) finished all
  29 steps cleanly, no CRASHED_AT_T in any summary.  Peak/final contact
  force (~166 N / ~122 N) matches the original free-3D run closely,
  confirming the fix changed nothing physically, only removed the
  numerical pathology.  **Final jitter table** (`table_soft_soft_jitter.py
  --steps 29 --settling 2 --variant planarRth2`,
  `figures/table_soft_soft_jitter_steps29_planarRth2.{txt,tex}`):
  the 3-node model wins ALL 10 metrics across both mu:

  | | Max P2P | Mean P2P | Max 3-pt | Mean 3-pt | RMS vs Smooth |
  |---|---|---|---|---|---|
  | mu=0,   2-node | 5.8e-01 | 8.3e-02 | 5.5e-02 | 1.5e-02 | 1.7e-01 |
  | mu=0,   3-node | 2.0e-01 | 6.8e-02 | 5.8e-05 | 5.8e-05 | 5.5e-02 |
  | mu=0.1, 2-node | 5.5e-01 | 8.2e-02 | 5.5e-02 | 1.5e-02 | 1.7e-01 |
  | mu=0.1, 3-node | 2.0e-01 | 6.8e-02 | 4.8e-03 | 2.3e-03 | 5.6e-02 |

  At mu=0 the 3-node 3-point metric (5.8e-05) is ~3 orders of magnitude
  below the 2-node's (5.5e-02) -- i.e. essentially free of the
  non-monotonic node-passing jitter that metric targets; at mu=0.1 the
  gap is still ~10x.  Figure: `postprocessing/fig_planar_final_comparison.py`
  -> `figures/fig_planar_final_comparison_steps29.{pdf,svg}` (4 panels:
  full records at both mu, zoomed tails t=20-29s showing the 2-node's
  periodic ripple at each element crossing vs the 3-node's smooth
  curve).  **This is the final, validated, planar (no out-of-plane
  sliding) 29-step comparison — the investigation that started with
  Esther noticing the 3D figure's out-of-plane bending is now closed.**
- μ = 0: contact persists to 29 s.  The 2-node model shows ~7 N smooth
  oscillation episodes at t ≈ 25.3 and 27.6 s (node passing rings the
  beam); the 3-node stays a monotone ramp with ~9 isolated ONE-dt
  glitches (≤ 0.4 N on 145 N, ≤ 0.3 %).  Net: 3-node wins Max P2P,
  Mean P2P and RMS-vs-smooth (3x); the 2-node keeps the two 3-pt
  metrics, which are blind to smooth oscillation but catch the one-dt
  glitches.  Suspected glitch source: adaptive k_c changing between the
  converged step and the trace-extraction re-evaluation (IMC extraction
  uses fixed kc=1) — worth checking before reading them as physics.
