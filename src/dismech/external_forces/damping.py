import typing
import scipy.sparse as sp
import numpy as np

from ..soft_robot import SoftRobot

def compute_damping_force(robot: SoftRobot, q: np.ndarray, u: np.ndarray) -> typing.Tuple[np.ndarray, typing.Union[np.ndarray, sp.csr_matrix]]:
    """
    Mass-proportional viscous damping (Rayleigh mass damping).

    Every translational DOF gets its own damping coefficient c_i = eta *
    m_i, where m_i is that DOF's entry in the lumped mass matrix (which
    already accounts for rod cross-section/density via the Voronoi
    reference length, and for shell thickness/density via the lumped
    face area -- see SoftRobot._get_mass_matrix). eta is therefore a
    damping RATE in 1/s, not a N.s/m coefficient: for a single damped
    oscillator m*x'' + c*x' + k*x = 0 with c = eta*m, this reduces to
    x'' + eta*x' + omega^2*x = 0 independent of m. Refining the mesh
    (more, lighter nodes covering the same physical body) no longer
    changes the total dissipation for a given velocity field, unlike a
    flat per-node coefficient.

    F = -eta * m_i * velocity_i
    J = -eta * m_i / dt * Identity

    Parameters
    ----------
    robot : object
        Must contain:
            - sim_params.dt
            - sim_params.sparse (bool)
            - n_dof
            - n_nodes
            - mass_matrix: (n_dof,) lumped mass per DOF
            - map_node_to_dof(i): returns [ix, iy, iz]
    q : (n_dof,) array
        Current DOF positions.
    u : (n_dof,) array
        Current DOF velocity.

    Returns
    -------
    Fd : (n_dof,) array
        Damping force.
    Jd : (n_dof, n_dof) array or sparse csr_matrix
        Damping Jacobian.
    """
    dt = robot.sim_params.dt
    eta = robot.env.eta   # damping rate, 1/s (mass-proportional coefficient)
    n_nodes = np.shape(robot.nodes)[0]
    n_dof = robot.n_dof

    # Translational DOF indices only (node positions), shape (n_nodes, 3)
    node_dof_indices = np.array([robot.map_node_to_dof(i) for i in range(n_nodes)])
    flat_dof_indices = node_dof_indices.reshape(-1).astype(int)

    c = eta * robot.mass_matrix[flat_dof_indices]  # per-DOF damping coefficient

    # Force
    Fd = np.zeros(n_dof)
    Fd[flat_dof_indices] = -c * u[flat_dof_indices]

    # Jacobian
    J_diag_vals = -c / dt
    if robot.sim_params.sparse:
        J_diag = np.zeros(n_dof)
        J_diag[flat_dof_indices] = J_diag_vals
        Jd = sp.diags((J_diag,), [0], shape=(n_dof, n_dof), format="csr")
    else:
        Jd = np.zeros((n_dof, n_dof))
        Jd[flat_dof_indices, flat_dof_indices] = J_diag_vals

    return Fd, Jd