import abc
import copy
import dataclasses
import typing

import scipy.sparse as sp
import numpy as np

from ..soft_robot import SoftRobot
from ..state import RobotState
from .. import profiling as _prof
from ..elastics import ElasticEnergy, StretchEnergy, HingeEnergy, BendEnergy, TriangleEnergy, TwistEnergy
from ..external_forces import compute_gravity_forces, compute_aerodynamic_forces_vectorized, compute_ground_contact, compute_ground_contact_friction, compute_rft, compute_damping_force, compute_surface_viscous_drag, compute_thrust_force_and_jacobian, add_point_forces
# from ..external_forces import predictor_step_for_ground_contact, corrector_step_for_ground_contact
from ..solvers import Solver, NumpySolver, PardisoSolver
from ..visualizer import Visualizer
from ..contact import (
    IMCEnergy,
    ShellContactEnergy,
    IMCFrictionEnergy,
    compute_circle_four_nodes_imc_weighted_forces_3d,
    compute_circle_four_nodes_imc_weighted_forces_3d_xy,
    compute_circle_node_bead_contact_forces,
    compute_self_contact_forces,
    compute_beam_to_beam_contact_forces,
    compute_eberly_3d_self_contact_forces,
    compute_eberly_3d_beam_to_beam_contact_forces,
    compute_eberly_3d_self_friction_forces,
    compute_eberly_3d_beam_to_beam_friction_forces,
    compute_eberly_3d_self_friction_setup,
    compute_eberly_3d_self_friction_apply,
    compute_eberly_3d_beam_to_beam_friction_setup,
    compute_eberly_3d_beam_to_beam_friction_apply,
    compute_eberly_3d_self_friction_pairs,
    compute_eberly_3d_self_friction_force,
    compute_eberly_3d_beam_to_beam_friction_pairs,
    compute_eberly_3d_beam_to_beam_friction_force,
    compute_eberly_3d_beam_to_beam_friction_force_with_jac,
)

_SOLVERS: typing.Dict[str, Solver] = {
    'np': NumpySolver, 'pardiso': PardisoSolver}


STRETCH = 'stretch'
HINGE = 'hinge'
MIDEDGE = 'triangle'
BEND = 'bend'
TWIST = 'twist'


class TimeStepper(metaclass=abc.ABCMeta):

    def __init__(self, robot: SoftRobot, min_force=1e-8, dtype=np.float64):
        self.robot = robot
        self._min_force = min_force
        self.original_dt = robot.sim_params.dt

        # Initialize elastics
        self.elastic_energies: typing.Dict[str, ElasticEnergy] = {}
        if robot.stretch_springs.N != 0:
            self.elastic_energies[STRETCH] = StretchEnergy(
                robot.stretch_springs, robot.state)
        if robot.hinge_springs.N != 0:
            self.elastic_energies[HINGE] = HingeEnergy(
                robot.hinge_springs, robot.state)
        if robot.triangle_springs:
            self.elastic_energies[MIDEDGE] = TriangleEnergy(
                robot.triangle_springs, robot.state)
        if robot.bend_springs.N != 0:
            self.elastic_energies[BEND] = BendEnergy(
                robot.bend_springs, robot.state
            )
            if not robot.sim_params.two_d_sim:   # if 3d
                self.elastic_energies[TWIST] = TwistEnergy(
                    robot.twist_springs, robot.state)

        if "selfContact" in robot.env.ext_force_list:
            self._contact_energy = IMCEnergy(robot.contact_pairs, robot.env.delta, robot.env.h, robot.env.imc_stiffness)

        if "shellContact" in robot.env.ext_force_list:
            self._contact_energy = ShellContactEnergy(robot.tri_contact_pairs, robot.env.delta, robot.env.h, robot.env.imc_stiffness)

        if "selfFriction" in robot.env.ext_force_list:
            self._imc_friction = IMCFrictionEnergy(robot.contact_pairs, robot.env.delta, robot.env.h, \
                                                   robot.sim_params.dt, robot.env.vel_tol, robot.env.mu, robot.env.imc_stiffness\
                                                   )

        # Set solver
        # TODO: figure out how to pass parameters
        self._solver = _SOLVERS.get(robot.sim_params.solver, NumpySolver)()

        # Simulate callbacks
        self.before_step = None

    def simulate(self, robot: SoftRobot = None, viz: Visualizer = None) -> typing.Tuple[typing.List[SoftRobot], typing.List[float], typing.List[float]]:
        robot = robot or self.robot
        steps = int(robot.sim_params.total_time / robot.sim_params.dt) + 1

        if viz is not None:
            viz.update(robot, 0)

        ret = []
        f_norms = []
        time_array = []
        i=0
        t=0.0
        time_array.append(t)
        ret.append(robot)
        while t < robot.sim_params.total_time:
            self._current_time = t

            # Handle user function
            if self.before_step is not None:
                robot = self.before_step(robot, i * robot.sim_params.dt)

            try:
                robot, f_norm = self.step(robot)
            except Exception as e:
                print("Error occurred during simulation step:", e)
                robot.sim_params.dt *= 0.1
                print("Reducing time step to:", robot.sim_params.dt)
                robot, f_norm = self.step(robot)

            # Update on step interval
            if viz is not None and i % robot.sim_params.plot_step == 0:
                viz.update(robot, t)
            if robot.sim_params.log_data and i % robot.sim_params.log_step == 0:
                ret.append(robot)
                f_norms.append(f_norm)
            i += 1
            t += robot.sim_params.dt
            time_array.append(t)

            # print current time
            print("current_time: ", t)
        return ret, time_array, f_norms

    def step(self, robot: SoftRobot = None, debug: bool = True) -> typing.Tuple[SoftRobot, float]:
        robot = robot or self.robot

        # Initialize iteration variables
        q = copy.deepcopy(robot.state.q)
        alpha = 1.0
        iteration = 1
        err_history = []
        solved = False

        # Preallocate matrices
        ndof_diag = np.arange(q.shape[0])

        while not solved:
            # Some integrators compute F and J not at q_{n+1} (midpoint)
            q_eval = self._compute_evaluation_position(robot, q)
            u_eval = self._compute_evaluation_velocity(robot, q)

            F = np.zeros(q.shape[0])

            if robot.sim_params.sparse:
                J = sp.csr_matrix(
                    (q.shape[0], q.shape[0]), dtype=np.float64)
            else:
                J = np.zeros((q.shape[0], q.shape[0]))

            # Inertial force vs equilibrium
            if not robot.sim_params.static_sim:
                inertial_force, inertial_jacobian = self._compute_inertial_force_and_jacobian(
                    robot, q)
                F += inertial_force

                if robot.sim_params.sparse:
                    J += sp.diags(inertial_jacobian, format='csr')
                else:
                    J[ndof_diag, ndof_diag] += inertial_jacobian

            F, J = self._compute_forces_and_jacobian(
                F, J, robot, q_eval, u_eval, iteration == 1)

            

            # Handle free DOF components
            f_free = F[robot.state.free_dof]
            if robot.sim_params.sparse:
                j_free = J[robot.state.free_dof,
                           :][:, robot.state.free_dof]
            else:
                j_free = J[np.ix_(
                    robot.state.free_dof, robot.state.free_dof)]

            # For debugging:
            # print("Determinant of Jacobian: ", np.linalg.det(j_free.toarray() if robot.sim_params.sparse else j_free))

            # Linear system solver
            if np.linalg.norm(f_free) < self._min_force:
                dq_free = np.zeros_like(f_free)
            else:
                with _prof.phase('solve'):
                    dq_free = self._solver.solve(j_free, f_free)

            # Adaptive damping and update
            if iteration > robot.sim_params.line_search_iters:
                if robot.sim_params.use_line_search:
                    alpha_orig = alpha
                    alpha = self._line_search(robot, q, dq_free, f_free, j_free)
                    # A returned step < 1 means the search backtracked.
                    if alpha < 1.0:
                        _prof.count('backtracks')

                    if alpha<=0.0:
                        alpha = self._adaptive_damping(alpha_orig)
                else:
                    alpha = self._adaptive_damping(alpha)
            dq_free *= alpha
            q[robot.state.free_dof] -= dq_free

            # Error and convergence
            err = np.linalg.norm(f_free)
            err_history.append(err)

            solved = self._converged(
                err, err_history, dq_free, iteration, robot)

            if debug:
                t_str = "t: {:.6f}, ".format(self._current_time) if hasattr(self, '_current_time') else ""
                print("{}iter: {}, error: {:.3f}".format(t_str, iteration, err))
            iteration += 1

        # Per-step Newton statistics (iteration counts one past the last).
        _prof.count('steps')
        _prof.count('newton_iters', iteration - 1)
        _prof.observe_max('newton_max', iteration - 1)

        if iteration >= robot.sim_params.max_iter:
            raise ValueError(
                "Iteration limit {} reached before convergence".format(robot.sim_params.max_iter))

        # Final update and return
        self.robot = self._finalize_update(robot, q)
        self.f_norm = np.linalg.norm(f_free)
        self.last_forces = F  # full force residual at convergence (reaction forces at fixed DOFs)
        return self.robot, self.f_norm

    @abc.abstractmethod
    def _compute_inertial_force_and_jacobian(self, robot: SoftRobot, q: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        pass

    def _compute_acceleration(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return np.zeros_like(q)

    def _compute_velocity(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return (q - robot.state.q) / robot.sim_params.dt

    def compute_total_elastic_energy(self, state: RobotState) -> np.ndarray:
        total = 0.0
        for energy in self.elastic_energies.values():
            total += energy.get_energy_linear_elastic(state)
        return np.array(total)

    def _compute_evaluation_position(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return q

    def _compute_evaluation_velocity(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return self._compute_velocity(robot, q)

    def _compute_forces_and_jacobian(self, forces, jacobian, robot: SoftRobot, q, u, first_iter=False):
        """ Computes forces and jacobian as sum of external and internal forces. """
        
        # Compute reference frames and material directors
        a1_iter, a2_iter = robot.compute_time_parallel(
            robot.state.a1, robot.state.q, q)
        m1, m2 = robot.compute_material_directors(q, a1_iter, a2_iter)
        ref_twist = robot.compute_reference_twist(
            robot.twist_springs, q, a1_iter, robot.state.ref_twist)
        tau = robot.update_pre_comp_shell(q)

        new_state = RobotState.init(
            q, a1_iter, a2_iter, m1, m2, ref_twist, tau)

        # Add elastic forces
        for energy in self.elastic_energies.values():
            F, J = energy.grad_hess_energy_linear_elastic(
                new_state, robot.sim_params.sparse)
            forces -= F
            jacobian -= J
        # Add external forces
        # TODO: Make this also a list
        if "gravity" in robot.env.ext_force_list:
            forces -= compute_gravity_forces(robot)
        # ignore for now
        if "aerodynamics" in robot.env.ext_force_list:
            F, J, = compute_aerodynamic_forces_vectorized(robot, q, u)
            forces -= F
            jacobian -= J  # FIXME: Sparse option
        if "damping" in robot.env.ext_force_list:
            F, J, = compute_damping_force(robot, q, u)
            forces -= F
            jacobian -= J  # FIXME: Sparse option
        if "hydrodynamics" in robot.env.ext_force_list:
            F, J, = compute_surface_viscous_drag(robot, q, u)
            forces -= F
            jacobian -= J  # FIXME: Sparse option
        if "thrust" in robot.env.ext_force_list:
            F, J, = compute_thrust_force_and_jacobian(robot, q, u)
            forces -= F
            jacobian -= J  # FIXME: Sparse option
        if "floorContact" in robot.env.ext_force_list:
            if "floorFriction" in robot.env.ext_force_list:
                F, J = compute_ground_contact_friction(robot, q, u)
            else:
                F, J = compute_ground_contact(robot, q)
            forces -= F
            jacobian -= J
        if "rft" in robot.env.ext_force_list:
            F, J = compute_rft(robot, q, u)
            forces -= F
            jacobian -= J
        if "selfContact" in robot.env.ext_force_list or "shellContact" in robot.env.ext_force_list:
            Fcon, Jcon = self._contact_energy.grad_hess_energy(new_state, robot, forces, first_iter)
            forces -= Fcon
            jacobian -= Jcon
        if "circleFourNodesIMCWeighted" in robot.env.ext_force_list:
            # 2D circle contact based on Esther's extended-arc IMC with 4-node weighting
            # in the (x,z) plane.
            F_con, J_con, _, _ = compute_circle_four_nodes_imc_weighted_forces_3d(
                robot,
                q,
                robot.env.circle_center,
                robot.env.circle_radius,
                robot.env.beam_radius,
                forces_3d=forces,
                first_iter=first_iter,
            )
            forces += F_con
            jacobian += J_con
        if "circleFourNodesIMCWeightedXY" in robot.env.ext_force_list:
            # 2D circle contact based on Esther's extended-arc IMC with 4-node weighting
            # in the (x,y) plane (used by contact_bending_test.py).
            F_con, J_con, _, _ = compute_circle_four_nodes_imc_weighted_forces_3d_xy(
                robot,
                q,
                robot.env.circle_center_3d,
                robot.env.circle_radius,
                robot.env.beam_radius,
                forces_3d=forces,
                first_iter=first_iter,
            )
            forces += F_con
            jacobian += J_con
        if "circleNodeBead" in robot.env.ext_force_list:
            # Simple bead-distance contact: each beam node is a sphere of
            # radius `beam_radius`, obstacle is a sphere of radius
            # `circle_radius` at `circle_center_3d`.  Adaptive k_c mirrors
            # the beamToBeamContactEberly3D logic.
            env = robot.env
            d_imc_nb = getattr(env, 'd_imc', 0.0008)
            if not hasattr(env, '_nodebead_kc'):
                env._nodebead_kc = env.imc_stiffness
                env._nodebead_active = False

            nb_candidate_set = False
            if first_iter:
                node_dof_nb = robot.node_dof_indices
                node_forces_nb = forces[node_dof_nb]
                max_force_nb = np.max(np.linalg.norm(node_forces_nb, axis=1))
                if max_force_nb >= 1e-9:
                    env._nodebead_candidate = max_force_nb * env.imc_stiffness
                    nb_candidate_set = True

            F_nb, J_nb, gap_min_nb = compute_circle_node_bead_contact_forces(
                robot, q,
                env.circle_center_3d,
                env.circle_radius,
                env.bead_radius,
                k_c=env._nodebead_kc,
                n_nodes_beam=env.n_nodes_beam_A,
                d_imc=d_imc_nb,
            )
            forces += F_nb
            jacobian += J_nb

            if first_iter and nb_candidate_set:
                ratio = 0.9
                if gap_min_nb < d_imc_nb:
                    env._nodebead_active = True
                    env._nodebead_kc = (ratio * env._nodebead_kc
                                       + (1 - ratio) * env._nodebead_candidate)
                elif env._nodebead_active:
                    env._nodebead_kc = (ratio * env._nodebead_kc
                                       + (1 - ratio) * env.imc_stiffness)
        if "selfContactExtendedArc" in robot.env.ext_force_list:
            # Extended-arc soft-to-soft self-contact (2D, x-y plane)
            n_nodes_sc = robot.n_nodes
            node_dof_sc = robot.node_dof_indices  # (n_nodes, 3)
            node_pos_sc = q[node_dof_sc]  # (n_nodes, 3)
            q2d_sc = np.stack([node_pos_sc[:, 0], node_pos_sc[:, 1]], axis=1).reshape(-1)

            # Adaptive k_c: scale contact stiffness with the force residual
            # so that contact forces are comparable to elastic/inertial forces.
            # Unlike the circle contact (which uses lagged smoothing), here we
            # apply the scaling immediately to prevent interpenetration.
            if first_iter:
                node_forces_sc = forces[node_dof_sc]  # (n_nodes, 3)
                max_force_sc = np.max(np.linalg.norm(node_forces_sc, axis=1))
                if max_force_sc > 1e-9:
                    k_c_sc = max_force_sc * robot.env.imc_stiffness
                else:
                    k_c_sc = robot.env.imc_stiffness
            else:
                # Reuse the value from the first iteration of this step
                k_c_sc = getattr(robot.env, '_self_contact_kc', robot.env.imc_stiffness)
            robot.env._self_contact_kc = k_c_sc

            _, f2d_sc, H2d_sc, all_gaps_sc, _ = compute_self_contact_forces(
                q2d_sc,
                robot.env.beam_radius,
                k_c=k_c_sc,
                min_segment_separation=robot.env.self_contact_min_sep,
            )

            # Embed 2D forces into 3D DOF vector (x, y components)
            for i_sc in range(n_nodes_sc):
                forces[node_dof_sc[i_sc, 0]] += f2d_sc[2 * i_sc]
                forces[node_dof_sc[i_sc, 1]] += f2d_sc[2 * i_sc + 1]

            # Embed 2D Hessian into 3D Jacobian (x, y blocks)
            for i_sc in range(n_nodes_sc):
                for j_sc in range(n_nodes_sc):
                    H_blk = H2d_sc[2 * i_sc: 2 * i_sc + 2, 2 * j_sc: 2 * j_sc + 2]
                    if not np.any(H_blk):
                        continue
                    ii_x, ii_y = node_dof_sc[i_sc, 0], node_dof_sc[i_sc, 1]
                    jj_x, jj_y = node_dof_sc[j_sc, 0], node_dof_sc[j_sc, 1]
                    jacobian[ii_x, jj_x] += H_blk[0, 0]
                    jacobian[ii_x, jj_y] += H_blk[0, 1]
                    jacobian[ii_y, jj_x] += H_blk[1, 0]
                    jacobian[ii_y, jj_y] += H_blk[1, 1]

        if "beamToBeamContactExtendedArc" in robot.env.ext_force_list:
            # Extended-arc beam-to-beam contact (2D, x-y plane)
            n_nodes_bb = robot.n_nodes
            node_dof_bb = robot.node_dof_indices  # (n_nodes, 3)
            node_pos_bb = q[node_dof_bb]  # (n_nodes, 3)
            q2d_bb = np.stack([node_pos_bb[:, 0], node_pos_bb[:, 1]], axis=1).reshape(-1)

            # Adaptive k_c
            if first_iter:
                node_forces_bb = forces[node_dof_bb]
                max_force_bb = np.max(np.linalg.norm(node_forces_bb, axis=1))
                if max_force_bb > 1e-9:
                    k_c_bb = max_force_bb * robot.env.imc_stiffness
                else:
                    k_c_bb = robot.env.imc_stiffness
            else:
                k_c_bb = getattr(robot.env, '_bb_contact_kc', robot.env.imc_stiffness)
            robot.env._bb_contact_kc = k_c_bb

            _, f2d_bb, H2d_bb, all_gaps_bb, _ = compute_beam_to_beam_contact_forces(
                q2d_bb,
                robot.env.n_nodes_beam_A,
                robot.env.beam_radius,
                k_c=k_c_bb,
                d_imc=robot.env.d_imc,
                kappa_threshold=robot.env.kappa_threshold,
            )

            # Embed 2D forces into 3D DOF vector (x, y components)
            for i_bb in range(n_nodes_bb):
                forces[node_dof_bb[i_bb, 0]] += f2d_bb[2 * i_bb]
                forces[node_dof_bb[i_bb, 1]] += f2d_bb[2 * i_bb + 1]

            # Embed 2D Hessian into 3D Jacobian (x, y blocks)
            for i_bb in range(n_nodes_bb):
                for j_bb in range(n_nodes_bb):
                    H_blk = H2d_bb[2 * i_bb: 2 * i_bb + 2, 2 * j_bb: 2 * j_bb + 2]
                    if not np.any(H_blk):
                        continue
                    ii_x, ii_y = node_dof_bb[i_bb, 0], node_dof_bb[i_bb, 1]
                    jj_x, jj_y = node_dof_bb[j_bb, 0], node_dof_bb[j_bb, 1]
                    jacobian[ii_x, jj_x] += H_blk[0, 0]
                    jacobian[ii_x, jj_y] += H_blk[0, 1]
                    jacobian[ii_y, jj_x] += H_blk[1, 0]
                    jacobian[ii_y, jj_y] += H_blk[1, 1]

        if "selfContactEberly3D" in robot.env.ext_force_list:
            # 3D self-contact using Eberly arc-to-arc distance (no 2D projection).
            # Adaptive k_c logic exactly mirrors the 2D circle contact in
            # circle_four_nodes_imc_weighted.py (_update/_finalize_contact_stiffness):
            #  * candidate = max_node_force * imc_stiffness, only when max_force >= 1e-9
            #  * k_c used for the force computation is the OLD smoothed value
            #  * after the force call, smooth UP (toward candidate) if contact is active,
            #    or DOWN (toward imc_stiffness) if it just deactivated; otherwise no change
            env = robot.env
            d_imc_sc = getattr(env, 'd_imc', 0.0008)
            if not hasattr(env, '_eberly_sc_kc'):
                env._eberly_sc_kc = env.imc_stiffness
                env._eberly_sc_active = False

            sc_candidate_set = False
            if first_iter:
                node_dof_eb = robot.node_dof_indices
                node_forces_eb = forces[node_dof_eb]
                max_force_eb = np.max(np.linalg.norm(node_forces_eb, axis=1))
                if max_force_eb >= 1e-9:
                    env._eberly_sc_candidate = max_force_eb * env.imc_stiffness
                    sc_candidate_set = True

            F_eb, J_eb, gap_min_eb, _ = compute_eberly_3d_self_contact_forces(
                robot, q, env.beam_radius, k_c=env._eberly_sc_kc,
                min_segment_separation=env.self_contact_min_sep,
                d_imc=d_imc_sc,
                # Opt-in 1/h stiffening, matching IMC's effective stiffness.
                h_scale=getattr(env, 'eberly_h_scale', False),
                # Opt-in line primitive for near-straight 3-node windows
                # (fitted arc radius above the threshold, or collinear).
                straight_R_threshold=getattr(
                    env, 'eberly_straight_R_threshold', None),
                # Opt-in crossfade to the two-node edge model where the
                # smooth conformity field detects extended contact.
                conforming_edge_blend=getattr(
                    env, 'eberly_conforming_edge_blend', False),
            )
            forces += F_eb
            jacobian += J_eb

            if first_iter and sc_candidate_set:
                ratio = 0.9
                kc_new = None
                if gap_min_eb < d_imc_sc:
                    env._eberly_sc_active = True
                    kc_new = (ratio * env._eberly_sc_kc
                              + (1 - ratio) * env._eberly_sc_candidate)
                elif env._eberly_sc_active:
                    kc_new = (ratio * env._eberly_sc_kc
                              + (1 - ratio) * env.imc_stiffness)
                if kc_new is not None:
                    # Opt-in governor on the adaptive stiffness.  The
                    # candidate (max nodal force × imc_stiffness) is noisy
                    # during contact transients — damping and elastic forces
                    # spike at snap-in — and letting kc chase it over orders
                    # of magnitude re-shapes the contact energy landscape
                    # faster than Newton can follow (50-iter failures in the
                    # 20260917 trefoil smoke test, kc swinging 100→2e5→3e4).
                    kc_rate = getattr(env, 'eberly_kc_rate', None)
                    if kc_rate is not None:
                        kc_new = min(max(kc_new, env._eberly_sc_kc / kc_rate),
                                     env._eberly_sc_kc * kc_rate)
                    kc_max = getattr(env, 'eberly_kc_max', None)
                    if kc_max is not None:
                        kc_new = min(kc_new, kc_max)
                    env._eberly_sc_kc = kc_new

        if "selfFrictionEberly3D" in robot.env.ext_force_list:
            # Coulomb friction.
            #
            # Velocity choice: we pass `robot.state.u` (END of previous dt-step)
            # rather than the in-iter `u = (q_iter - q_prev_step)/dt`.  At
            # first_iter the in-iter u is identically zero (q_iter == q_prev),
            # so a fully-implicit-in-u friction is identically zero on the
            # first Newton step.  With strong contact, the Newton step from
            # iter 1 -> iter 2 is then computed WITHOUT friction and overshoots
            # the true equilibrium by exactly the friction force; at iter 2
            # the in-iter velocity reflects that overshoot and the friction
            # force explodes, causing Newton to diverge.  Using the previous
            # converged velocity keeps friction non-zero from iter 1 and
            # essentially constant across iters of a dt-step (quasi-static
            # loading means u is slowly-varying), which is what Newton needs
            # to find equilibrium without chattering.  The pair list is
            # frozen at first_iter (matching IMC).
            env = robot.env
            d_imc_fric_sc = getattr(env, 'd_imc', 0.0008)
            if first_iter:
                env._eberly_sc_fric_pairs = compute_eberly_3d_self_friction_pairs(
                    robot, q, env.beam_radius,
                    min_segment_separation=env.self_contact_min_sep,
                    d_imc=d_imc_fric_sc,
                )
            k_c_fric_sc = getattr(env, '_eberly_sc_kc', env.imc_stiffness)
            F_fric_sc = compute_eberly_3d_self_friction_force(
                robot, q, robot.state.u, env._eberly_sc_fric_pairs,
                env.beam_radius, mu=env.mu, vel_tol=env.vel_tol,
                k_c=k_c_fric_sc, d_imc=d_imc_fric_sc,
                straight_R_threshold=getattr(
                    env, 'eberly_straight_R_threshold', None),
                conforming_edge_blend=getattr(
                    env, 'eberly_conforming_edge_blend', False),
            )
            forces -= F_fric_sc

        if "beamToBeamContactEberly3D" in robot.env.ext_force_list:
            # 3D beam-to-beam contact using Eberly arc-to-arc distance.
            # Adaptive k_c logic exactly mirrors the 2D circle contact in
            # circle_four_nodes_imc_weighted.py (_update/_finalize_contact_stiffness).
            env = robot.env
            d_imc_bb = getattr(env, 'd_imc', 0.0008)
            if not hasattr(env, '_eberly_bb_kc'):
                env._eberly_bb_kc = env.imc_stiffness
                env._eberly_bb_active = False

            bb_candidate_set = False
            if first_iter:
                node_dof_eb2 = robot.node_dof_indices
                node_forces_eb2 = forces[node_dof_eb2]
                max_force_eb2 = np.max(np.linalg.norm(node_forces_eb2, axis=1))
                if max_force_eb2 >= 1e-9:
                    env._eberly_bb_candidate = max_force_eb2 * env.imc_stiffness
                    bb_candidate_set = True

            F_eb2, J_eb2, gap_min_eb2, _ = compute_eberly_3d_beam_to_beam_contact_forces(
                robot, q, env.beam_radius, k_c=env._eberly_bb_kc,
                n_nodes_beam_A=env.n_nodes_beam_A,
                d_imc=d_imc_bb,
                # Opt-in: pair-level line fallback when either arc's fitted
                # radius exceeds `env.eberly_straight_R_threshold`. Only
                # `double_spiral_test.py` sets this attribute right now.
                straight_R_threshold=getattr(env, 'eberly_straight_R_threshold', None),
                # Opt-in 1/h stiffening, matching IMC's effective stiffness.
                h_scale=getattr(env, 'eberly_h_scale', False),
            )
            forces += F_eb2
            jacobian += J_eb2

            if first_iter and bb_candidate_set:
                ratio = 0.9
                if gap_min_eb2 < d_imc_bb:
                    env._eberly_bb_active = True
                    env._eberly_bb_kc = (ratio * env._eberly_bb_kc
                                         + (1 - ratio) * env._eberly_bb_candidate)
                elif env._eberly_bb_active:
                    env._eberly_bb_kc = (ratio * env._eberly_bb_kc
                                         + (1 - ratio) * env.imc_stiffness)

        if "beamToBeamFrictionEberly3D" in robot.env.ext_force_list:
            # See comment in selfFrictionEberly3D branch for the velocity
            # choice and pair-list freezing rationale.
            env = robot.env
            d_imc_fric_bb = getattr(env, 'd_imc', 0.0008)
            if first_iter:
                env._eberly_bb_fric_pairs = compute_eberly_3d_beam_to_beam_friction_pairs(
                    robot, q, env.beam_radius,
                    n_nodes_beam_A=env.n_nodes_beam_A,
                    d_imc=d_imc_fric_bb,
                )
            k_c_fric_bb = getattr(env, '_eberly_bb_kc', env.imc_stiffness)

            # Velocity choice for the friction force.  Three modes:
            #   'in_iter' (default): u = (q_iter - q_prev)/dt at every Newton
            #       iter.  At iter 1 q_iter == q_prev so u = 0 and friction
            #       is identically zero — Newton's first step balances only
            #       elastic+contact, then later iters bring in the friction
            #       residual.  Empirically this gives the best F_N at mu=0.8.
            #   'state_u': u = robot.state.u (END of previous dt-step) at every
            #       iter.  Pure semi-explicit; legacy non-Jacobian path.
            #   'hybrid' (Option C): state.u at first iter, in-iter at later
            #       iters.  Intended to avoid the iter-1 "zero friction" issue
            #       but couples friction force into the adaptive k_c update
            #       at first iter, inflating k_c at high mu.
            vel_mode = getattr(env, "eberly_friction_velocity_mode", "in_iter")
            if vel_mode == "state_u":
                u_for_fric = robot.state.u
            elif vel_mode == "hybrid":
                u_for_fric = robot.state.u if first_iter else u
            else:   # 'in_iter' default
                u_for_fric = u

            # Friction law params (env overrides; defaults preserve the smooth
            # Coulomb gate behavior so existing scripts are unaffected).
            friction_law = getattr(env, "friction_law", "smooth_coulomb")
            nh_rho       = getattr(env, "norton_hoff_rho", 0.1)
            nh_v_ref     = getattr(env, "norton_hoff_v_ref", 1.0)

            if getattr(env, "eberly_friction_jacobian", False):
                F_fric_bb, J_fric_bb = compute_eberly_3d_beam_to_beam_friction_force_with_jac(
                    robot, q, u_for_fric, env._eberly_bb_fric_pairs,
                    env.beam_radius, mu=env.mu, vel_tol=env.vel_tol,
                    dt=robot.sim_params.dt,
                    k_c=k_c_fric_bb, d_imc=d_imc_fric_bb,
                    friction_law=friction_law,
                    nh_rho=nh_rho, nh_v_ref=nh_v_ref,
                )
                forces -= F_fric_bb
                # Skip the velocity-Jacobian add when the velocity we used
                # at this iter does NOT depend on q (then ∂F/∂q via velocity
                # is exactly zero — adding a non-zero matrix would corrupt
                # Newton's tangent).
                if not (vel_mode == "state_u"
                        or (vel_mode == "hybrid" and first_iter)):
                    jacobian -= J_fric_bb
            else:
                F_fric_bb = compute_eberly_3d_beam_to_beam_friction_force(
                    robot, q, u_for_fric, env._eberly_bb_fric_pairs,
                    env.beam_radius, mu=env.mu, vel_tol=env.vel_tol,
                    k_c=k_c_fric_bb, d_imc=d_imc_fric_bb,
                    friction_law=friction_law,
                    nh_rho=nh_rho, nh_v_ref=nh_v_ref,
                )
                forces -= F_fric_bb

        if "selfFriction" in robot.env.ext_force_list:
            # BUG FIX: RobotState.init() sets u=zeros, so without this patch
            # new_state.u is always zero and IMCFrictionEnergy.grad_friction
            # returns zero force (verified in a standalone test).  Replace
            # new_state with one carrying the in-iter velocity `u` (which is
            # u_eval = (q - robot.state.q)/dt for implicit Euler).
            new_state_with_u = dataclasses.replace(new_state, u=u)
            # Opt-in: when env.imc_friction_jacobian is True, also assemble
            # the velocity-Jacobian block (∂F_fric/∂v · 1/dt) into the
            # global tangent.  Default keeps the legacy force-only path.
            if getattr(robot.env, "imc_friction_jacobian", False):
                F, J = self._imc_friction.grad_hess_friction(
                    new_state_with_u, robot, forces, first_iter)
                forces -= F
                jacobian -= J
            else:
                F = self._imc_friction.grad_friction(
                    new_state_with_u, robot, forces, first_iter)
                forces -= F
        # NOTE: gravity is already applied once, right after the elastic
        # forces (top of this function). A second identical block here used
        # to double every gravity load -- do not re-add it.
        if "pointForces" in robot.env.ext_force_list:
            forces -= add_point_forces(robot, robot.env.point_force_node_indices, robot.env.point_force_vectors)

        return forces, jacobian

    def _converged(self,
                   err: float,
                   err_history: typing.List[float],
                   dq: np.ndarray,
                   iteration: int,
                   robot: SoftRobot):
        """ Check all convergence criteria """
        #print(err)
        disp_converged = np.max(np.abs(dq)) / \
            robot.sim_params.dt < robot.sim_params.dtol
        force_converged = err < robot.sim_params.tol
        relative_converged = err < err_history[0] * robot.sim_params.ftol
        iteration_limit = iteration >= robot.sim_params.max_iter

        return any([force_converged, relative_converged, disp_converged, iteration_limit])

    def _adaptive_damping(self, alpha):
        """ Adaptive damping based on the current alpha value. """
        return max(alpha * 0.9, 0.1)

    def _line_search(self, robot, q, dq, F, J, m1=0.1, m2=0.9, alpha_low=0.0, alpha_high=1.0):
        assert F.shape[0] == len(robot.state.free_dof), "F must be reduced to free DOFs"
        d0 = -np.dot(F, J @ dq)  # Directional derivative (should be negative)

        if d0 >= 0:
            # Not a descent direction, don't take any step
            return 0.0

        alpha = alpha_high
        alpha_min = 1e-6
        alpha_max = 2.0
        max_iter = 100
        tol = 1e-6
        iteration = 0
        success = False

        ndof = q.shape[0]
        free_dof = robot.state.free_dof

        while not success:
            # Build full dq vector
            dq_full = np.zeros_like(q)
            dq_full[free_dof] = dq
            q_new = q - alpha * dq_full

            # Compute updated state
            q_eval = self._compute_evaluation_position(robot, q_new)
            u_eval = self._compute_evaluation_velocity(robot, q_new)

            # Recompute force at new configuration
            F_iter = np.zeros_like(q)
            if robot.sim_params.sparse:
                J_iter = sp.csr_matrix((ndof, ndof), dtype=np.float64)
            else:
                J_iter = np.zeros((ndof, ndof))

            if not robot.sim_params.static_sim:
                inertial_force, _ = self._compute_inertial_force_and_jacobian(robot, q_new)
                F_iter += inertial_force
                # Optional: add inertia Jacobian if needed

            F_new, _ = self._compute_forces_and_jacobian(F_iter, J_iter, robot, q_eval, u_eval)

            # Evaluate Wolfe conditions (on free DOFs only)
            F_norm_sq_old = 0.5 * np.linalg.norm(F)**2
            F_norm_sq_new = 0.5 * np.linalg.norm(F_new[free_dof])**2
            lhs = F_norm_sq_new - F_norm_sq_old
            rhs_low = alpha * m2 * d0
            rhs_high = alpha * m1 * d0

            if rhs_low <= lhs <= rhs_high:
                success = True
            elif lhs < rhs_low:
                alpha_low = alpha
            else:
                alpha_high = alpha

            # Update alpha
            if alpha_high < alpha_max:
                alpha = 0.5 * (alpha_low + alpha_high)
            else:
                alpha = 10 * alpha

            iteration += 1

            if (alpha < alpha_min) or (iteration > max_iter) or ((alpha_high - alpha_low) < tol):
                # Line search failed to find suitable step
                return 0.0

        return alpha


    def _finalize_update(self, robot: SoftRobot, q):
        u = self._compute_velocity(robot, q)
        a = self._compute_acceleration(robot, q)
        a1, a2 = robot.compute_time_parallel(robot.state.a1, robot.state.q, q)
        m1, m2 = robot.compute_material_directors(q, a1, a2)
        ref_twist = robot.compute_reference_twist(
            robot.twist_springs, q, a1, robot.state.ref_twist)
        return robot.update(q=q, u=u, a=a, a1=a1, a2=a2, m1=m1, m2=m2, ref_twist=ref_twist)
