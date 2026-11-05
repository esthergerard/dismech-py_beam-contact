"""
Simple bead-distance contact between a deformable beam (treated as a chain of
spheres of radius `bead_radius`) and a rigid circular obstacle (treated as a
single sphere of radius `circle_radius` centred at `circle_center`).

Geometric gap for each beam node `i`:

    gap_i = || X_i - C || - circle_radius - bead_radius

The natural choice for ``bead_radius`` is HALF the inter-node spacing
(``bead_radius = L / (2 (N - 1))``).  With that choice adjacent beads are
tangent in the undeformed configuration: no overlap, no gap.  If
``bead_radius`` is bigger than the half-spacing adjacent beads penetrate
each other and several beads can simultaneously sit inside the activation
band of the obstacle, which destabilises the implicit solver.

The penalty energy is the same IMC law used by the other contact models
(``circle_four_nodes_imc_weighted`` / ``eberly_contact_3d``), evaluated at
``delta_c = gap + bead_radius`` so that the activation threshold and the
``d_imc`` regularisation band are consistent with those models.

The 1/bead_radius normalisation factor applied to F and J mirrors the
convention used by IMCEnergy / Eberly so that contact stiffness ``k_c`` has
the same magnitude across all contact laws.
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..soft_robot import SoftRobot

from .circle_four_nodes_imc_weighted import (
    compute_dE_IMC_ddelta_c,
    compute_d2E_IMC_ddelta_c,
)


def compute_circle_node_bead_contact_forces(
    robot: "SoftRobot",
    q: np.ndarray,
    circle_center: np.ndarray,
    circle_radius: float,
    bead_radius: float,
    k_c: float = 1.0,
    n_nodes_beam: int = None,
    d_imc: float = 0.0008,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Parameters
    ----------
    robot : SoftRobot
        Used only for ``node_dof_indices`` and ``n_nodes``.
    q : (ndof,) full DOF vector.
    circle_center : (3,) world-frame position of the obstacle centre.
    circle_radius : obstacle radius.
    bead_radius : radius of each bead.  The natural choice is half the
        inter-node spacing,  ``L / (2 (N - 1))``.  This is INDEPENDENT of
        the rod cross-section radius.
    k_c : contact stiffness multiplier.
    n_nodes_beam : number of nodes in the deformable beam.  Defaults to
        ``robot.n_nodes``.  Nodes 0..n_nodes_beam-1 are checked against the
        obstacle (this matches the Eberly beam-to-beam convention).
    d_imc : IMC regularisation width.

    Returns
    -------
    F : (ndof,) gradient of the contact energy (use ``forces += F``).
    J : (ndof, ndof) Hessian of the contact energy.
    gap_min : minimum gap across all checked nodes (``+inf`` if none).
    """
    node_dof = robot.node_dof_indices
    n_nodes = robot.n_nodes
    ndof = q.shape[0]

    if n_nodes_beam is None:
        n_nodes_beam = n_nodes

    F = np.zeros(ndof)
    J = np.zeros((ndof, ndof))

    K_imc = 15.0 / d_imc
    activation_gap = d_imc

    C = np.asarray(circle_center, dtype=float).reshape(3)

    gap_min = np.inf

    for i in range(n_nodes_beam):
        dof_idx = node_dof[i]
        X = q[dof_idx]
        D = X - C
        Delta = float(np.linalg.norm(D))
        if Delta < 1e-12:
            continue

        gap = Delta - circle_radius - bead_radius
        if gap < gap_min:
            gap_min = gap

        if gap > activation_gap + 1e-12:
            continue

        delta_c = gap + bead_radius

        dE = float(compute_dE_IMC_ddelta_c(
            np.asarray(delta_c), bead_radius, d_imc, K_imc))
        d2E = float(compute_d2E_IMC_ddelta_c(
            np.asarray(delta_c), bead_radius, d_imc, K_imc))

        # grad(gap) w.r.t. X_i  = D / Delta = unit vector from C to X
        unit = D / Delta

        # Force contribution: dE/d_delta_c * d_delta_c/dX_i  (d_delta_c == d_gap)
        F[dof_idx] += dE * unit

        # Hess(gap) w.r.t. X_i = (I - unit unit^T) / Delta
        H_gap = (np.eye(3) - np.outer(unit, unit)) / Delta
        H_local = d2E * np.outer(unit, unit) + dE * H_gap
        J[np.ix_(dof_idx, dof_idx)] += H_local

    # 1/bead_radius normalisation matches IMCEnergy / Eberly so that k_c
    # has comparable magnitudes across contact laws.
    scale = 1.0 / bead_radius
    F *= k_c * scale
    J *= k_c * scale * scale

    return F, J, gap_min
