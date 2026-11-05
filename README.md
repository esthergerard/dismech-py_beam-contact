# dismech-py_beam-contact

Companion code for **"Modeling contact in elastic rods: contact
geometry for jitter-free simulation"** (Gérard, Jawed, Boucard, Louf).
This repo contains the three-node arc-based contact model implemented
in DisMech (a discrete-differential-geometry rod/shell simulator) and
everything needed to reproduce the paper's three benchmarks and
appendix material: a beam bent against a rigid obstacle, two orthogonal
elastic beams in frictional contact, and the pull-out of a strand from
a woven net.

The contact model itself lives in `src/dismech/contact/` (see in
particular `eberly_contact_3d.py` for the arc-to-arc distance query and
`circle_four_nodes_imc_weighted.py` for the blended-weight contact
energy). It is a drop-in geometric replacement for the usual two-node
(segment) contact representation and works with any energy-based
contact law — nothing about the constitutive model (here, the Implicit
Contact Model, IMC) had to change.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate           # .venv\Scripts\activate on Windows
pip install -r requirements.txt     # pinned environment used for the paper's results
pip install -e .                    # editable install of this package
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e .                 # creates .venv automatically
```

Verify the install:

```bash
pytest tests/
```

## Reproducing the paper

See `benchmarks/PAPER_MAP.md` for the exact figure/table -> script ->
data mapping, and `benchmarks/paper_reproducibility_params.md` for
every simulation parameter (material, time step, contact stiffness,
etc.) with its source location.

| benchmark | paper section | folder |
|---|---|---|
| Beam vs. rigid circular obstacle | Sec. 3 | `benchmarks/three-pts_bend_test/` |
| Two orthogonal elastic beams (Zavarise & Wriggers) | Sec. 4 | `benchmarks/two_orthogonal_beams/` |
| Woven net strand pull-out | Sec. 5 | `benchmarks/net_pullout/` |
| Computational cost | Sec. 4.5 | `benchmarks/computational_cost/` |
| Appendix A toy cases | App. A | `benchmarks/toy_cases_appendix/` |

Each benchmark folder has its own `NOTES.md`/README with the exact
commands to regenerate its data and figures from scratch; most figure
scripts can also just be re-run directly against the data already
shipped in the repo (no simulation rerun needed):

```bash
cd benchmarks/three-pts_bend_test/postprocessing
python3 fig6_jitter_metrics.py   # reproduces Table 1
```

## Features of the underlying simulator

- 3D [discrete elastic rod](https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf) stretching, bending and twisting.
- 3D discrete elastic shell hinge and mid-edge bending.
- Pytest numerical validation suite (compared against [Dismech Matlab](https://github.com/StructuresComp/dismech-matlab)).
- Implicit integration schemes (Euler and Newmark-beta).
- Dense and sparse ([PyPardiso](https://pypi.org/project/pypardiso/)) solvers.
- [IMC](https://arxiv.org/abs/2205.10309) rod contact and friction modeling, plus the three-node arc-based geometric model of this paper.
- Shell contact and friction modeling.

This repo is derived from [dismech-python](https://github.com/StructuresComp/dismech-rods) / Py-DiSMech; see the Citation section below for the underlying framework.

## Citation

If this contact model helped your research, please cite:
```
@article{gerard_contact_geometry,
  title   = {Modeling contact in elastic rods: contact geometry for jitter-free simulation},
  author  = {G{\'e}rard, Esther and Jawed, M. Khalid and Boucard, Pierre-Alain and Louf, Fran{\c c}ois},
  note    = {Manuscript},
}
```

The underlying DisMech framework:
```
@misc{lahoti2025pydismechscalableefficientframework,
      title={Py-DiSMech: A Scalable and Efficient Framework for Discrete Differential Geometry-Based Modeling and Control of Soft Robots},
      author={Radha Lahoti and Ryan Chaiyakul and M. Khalid Jawed},
      year={2025},
      eprint={2512.09911},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.09911},
}
```
