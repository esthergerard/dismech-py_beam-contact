#!/usr/bin/env python3
"""
Three-point bending test with contact — two-node (segment) variant.

Identical setup to contact_bending_test.py (the three-node arc model)
except for the contact geometry: the circular obstacle is discretized
into a closed polygon of `n_circle_nodes` fixed nodes (matching the
paper's "support is itself discretized, with 200 nodes around the
circle", Sec. 3.1), and contact is resolved edge-to-edge between beam
segments and circle segments using dismech's existing two-node IMC
self-contact machinery (`env.add_force('selfContact', ...)`) — the
same contact law already used for the 2-node baseline in
two_orthogonal_beams/ and net_pullout/. No new contact-energy code is
introduced; this script only wires together existing dismech pieces
(ContactEnergy/IMCEnergy via 'selfContact', the cross-beam-only pair
construction already used for two-beam contact) around the circle
geometry builder that was already present, but unused, in
contact_bending_test.py.

Radius bookkeeping: 'selfContact' assumes BOTH contacting bodies have
the same edge radius `h` (gap = centerline distance - 2h), which is
exactly right for beam-to-beam contact between two rods of equal
radius. Here the second "body" is a zero-thickness rigid curve (the
true geometric circle, radius `circle_radius`), not a second rod of
radius `beam_radius`. To get the correct physical gap
(= ||X - C|| - circle_radius - beam_radius) through the symmetric
formula, the discretized circle is placed at radius
`circle_radius - beam_radius` instead of `circle_radius`: then
centerline-to-centerline distance minus `2*beam_radius` equals the
true surface-to-surface gap. See `create_circle_geometry` below.

All other simulation parameters (geometry, material, dt, d_imc,
damping, clamping, imposed velocity, Newton tolerances) are unchanged
from contact_bending_test.py, so the three contact models are compared
on identical physics with only the contact geometry varying.
"""

import argparse
from pathlib import Path
import numpy as np
import dismech

# ============================================================================
# PARAMETERS
# ============================================================================
beam_length = 1.0  # m
beam_y = -0.202  # m
beam_radius = 0.001  # m
n_beam_nodes = 10  # default number of nodes along the beam

parser = argparse.ArgumentParser(
    description="Three-point bending test with contact (2-node / segment model)")
parser.add_argument("n_nodes", nargs="?", type=int, default=n_beam_nodes,
                     help="Number of beam nodes (default: %(default)s)")
parser.add_argument("--restart", action="store_true",
                     help="Restart from the last saved snapshot in the coordinates file")
parser.add_argument("--n-circle-nodes", type=int, default=200,
                     help="Number of nodes discretizing the circular obstacle "
                          "(default: %(default)s, matching the paper's Sec. 3.1)")
cli_args = parser.parse_args()
n_beam_nodes = cli_args.n_nodes
restart_mode = cli_args.restart
n_circle_nodes = cli_args.n_circle_nodes

# Circle obstacle parameters
circle_center = np.array([0.8, 0.0, 0.0])  # m
circle_radius = 0.2  # m

# Material properties
E = 210e8  # Young's modulus [Pa]
nu = 0.3  # Poisson's ratio
density = 7850  # Steel density [kg/m^3]

# Quasi-static / inextensibility scaling factors (see contact_bending_test.py)
k_inext = 1e3
m_quasi = 0.01

# Velocity control parameters
imposed_velocity = 1.0e-3  # [m/s]
max_displacement = 1.2  # [m]

# Simulation parameters
dt = 1.0e-3  # [s]
total_time = 1000.0  # [s]
save_interval = 0.01  # [s]

# Contact parameters (shared with the 1-node and 3-node scripts)
d_imc = 0.0008  # IMC regularisation width [m], = delta in the paper's Table

# ============================================================================
# GEOMETRY CREATION
# ============================================================================

def create_beam_geometry(length, n_nodes, y_pos, kappa0=-1e-3):
    """Same construction as contact_bending_test.py: constant-curvature arc
    seeded with a small kappa0 to avoid the straight-edge degeneracy."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2 to create a 1D beam mesh")

    s = np.linspace(0.0, length, n_nodes)
    theta = kappa0 * s
    ds = np.diff(s)
    dx = ds * np.cos(0.5 * (theta[:-1] + theta[1:]))
    dy = ds * np.sin(0.5 * (theta[:-1] + theta[1:]))

    nodes = np.zeros((n_nodes, 3))
    nodes[0, 0] = 0.0
    nodes[0, 1] = y_pos
    nodes[1:, 0] = np.cumsum(dx)
    nodes[1:, 1] = y_pos + np.cumsum(dy)

    edges = np.array([[i, i + 1] for i in range(n_nodes - 1)], dtype=int)
    face_nodes = np.empty((0, 3), dtype=int)
    return dismech.Geometry(nodes, edges, face_nodes)


def create_circle_geometry(center, radius, n_nodes):
    """Closed polygon of `n_nodes` approximating the circular obstacle.

    `radius` here is the DISCRETIZATION radius, already offset by
    -beam_radius so that the symmetric two-node contact formula
    (gap = centerline distance - 2*beam_radius) reproduces the true
    physical gap (= centerline distance to the true circle - circle_radius
    - beam_radius). Callers should pass `circle_radius - beam_radius`,
    not `circle_radius` itself.
    """
    nodes = np.zeros((n_nodes, 3))
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    nodes[:, 0] = center[0] + radius * np.cos(angles)
    nodes[:, 1] = center[1] + radius * np.sin(angles)
    nodes[:, 2] = center[2]

    edges = np.array([[i, (i + 1) % n_nodes] for i in range(n_nodes)], dtype=int)
    face_nodes = np.empty((0, 3), dtype=int)
    return dismech.Geometry(nodes, edges, face_nodes)


def combine_geometries(geo1, geo2):
    """Stack two geometries into one (beam first, then circle), so that
    edge index 0..n_edges_beam-1 belongs to the beam and the rest to the
    circle — the ordering `sim_params.n_edges_beam_A` relies on."""
    n1 = geo1.nodes.shape[0]
    combined_nodes = np.vstack([geo1.nodes, geo2.nodes])
    edges2_offset = geo2.edges + n1
    combined_edges = np.vstack([geo1.edges, edges2_offset])
    face_nodes = np.empty((0, 3), dtype=int)
    return dismech.Geometry(combined_nodes, combined_edges, face_nodes)


# ============================================================================
# RESTART UTILITIES
# ============================================================================

def load_last_snapshot(coords_file, n_nodes):
    """Identical to contact_bending_test.py's restart reader, but only the
    beam's own nodes are stored in the coords file (n_nodes = beam nodes)."""
    last_time = None
    coords = np.zeros((n_nodes, 3))
    lines_in_snapshot = 0

    with open(coords_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            t = float(parts[0])
            node_id = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])

            if last_time is None or t != last_time:
                last_time = t
                coords = np.zeros((n_nodes, 3))
                lines_in_snapshot = 0

            coords[node_id] = [x, y, z]
            lines_in_snapshot += 1

    if last_time is None or lines_in_snapshot < n_nodes:
        raise RuntimeError(
            f"Could not find a complete snapshot in '{coords_file}' "
            f"(expected {n_nodes} nodes per snapshot)."
        )

    print(f"  Loaded restart snapshot at t = {last_time:.6f} s from '{coords_file}'")
    return last_time, coords


# ============================================================================
# CROSS-SECTION PROPERTIES
# ============================================================================

def calculate_cross_section_properties(r):
    A = np.pi * r**2
    I = np.pi * r**4 / 4
    J = np.pi * r**4 / 2
    return A, I, J


# ============================================================================
# SIMULATION
# ============================================================================

def run_contact_bending_test_2node(restart=False):
    """Run the contact bending test with the two-node (segment) contact model."""
    print("=" * 70)
    print("THREE-POINT BENDING TEST WITH CONTACT (2-node / segment model)")
    if restart:
        print(">>> RESTART MODE <<<")
    print("=" * 70)

    beam_geo = create_beam_geometry(beam_length, n_beam_nodes, beam_y)
    n_beam_nodes_total = beam_geo.nodes.shape[0]
    beam_node_indices = np.arange(n_beam_nodes_total)
    n_beam_edges = n_beam_nodes_total - 1
    print(f"Created beam geometry: {n_beam_nodes_total} nodes, {n_beam_edges} edges "
          f"(uniform mesh; fixed clamp nodes [0, 1])")

    # Discretized circle, placed at radius - beam_radius (see module docstring).
    circle_disc_radius = circle_radius - beam_radius
    circle_geo = create_circle_geometry(circle_center, circle_disc_radius, n_circle_nodes)
    circle_node_indices = n_beam_nodes_total + np.arange(n_circle_nodes)
    print(f"Created circle geometry: {n_circle_nodes} nodes at discretization "
          f"radius {circle_disc_radius:.6f} m (true obstacle radius {circle_radius} m, "
          f"offset by beam_radius = {beam_radius} m)")

    combined_geo = combine_geometries(beam_geo, circle_geo)

    A, I, J = calculate_cross_section_properties(beam_radius)
    A_eff = k_inext * A
    I_eff = I
    J_eff = J
    density_eff = (density / k_inext) * m_quasi

    geom = dismech.GeomParams(
        rod_r0=beam_radius, shell_h=0,
        axs=A_eff, ixs1=I_eff, ixs2=I_eff, jxs=J_eff,
    )
    material = dismech.Material(
        density=density_eff,
        youngs_rod=E, youngs_shell=0,
        poisson_rod=nu, poisson_shell=0,
    )

    log_step = max(1, int(round(save_interval / dt)))
    effective_save_interval = log_step * dt
    print(f"Saving state every {effective_save_interval:.4f} s (log_step={log_step})")

    # cross_beam_only_contact=True restricts contact pairs to beam-vs-circle
    # edges only: no beam self-pairs (not needed here), and critically no
    # circle self-pairs (which would otherwise fire at every pair of
    # adjacent — and many non-adjacent — segments around the tight loop).
    sim_params = dismech.SimParams(
        static_sim=False,
        two_d_sim=True,
        use_mid_edge=False,
        use_line_search=False,
        show_floor=False,
        log_data=True,
        log_step=log_step,
        dt=dt,
        max_iter=50,
        total_time=total_time,
        plot_step=0,
        tol=1e-5,
        ftol=1e-5,
        dtol=1e-5,
        n_edges_beam_A=n_beam_edges,
        cross_beam_only_contact=True,
    )

    env = dismech.Environment()
    env.add_force(
        'selfContact',
        delta=d_imc,
        h=beam_radius,
        kc=1.0,
    )
    env.add_force('damping', eta=1.0)  # Damping for quasi-static

    robot = dismech.SoftRobot(geom, material, combined_geo, sim_params, env)
    print("Created SoftRobot (beam + discretized circle)")

    left_end_nodes = np.array([0, 1])
    robot = robot.fix_nodes(left_end_nodes, fix_edges=False)
    print(f"Fixed left end nodes {left_end_nodes} for position control")

    robot = robot.fix_nodes(circle_node_indices, fix_edges=False)
    print(f"Fixed all {n_circle_nodes} circle nodes (rigid obstacle)")

    stepper = dismech.ImplicitEulerTimeStepper(robot)

    out_dir = Path(__file__).resolve().parent / "contact_output_files_2node"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_coords_txt = out_dir / f"reg_beam_coords_N{n_beam_nodes}_2node.txt"
    output_forces_txt = out_dir / f"reg_contact_forces_N{n_beam_nodes}_2node.txt"
    output_summary_txt = out_dir / f"reg_summary_N{n_beam_nodes}_2node.txt"

    restart_time = 0.0
    restart_step = 0

    if restart:
        restart_time, restart_coords = load_last_snapshot(
            output_coords_txt, n_beam_nodes_total
        )
        restart_step = int(round(restart_time / dt))

        q_restart = np.copy(robot.state.q)
        for ni in range(n_beam_nodes_total):
            dofs = robot.map_node_to_dof(np.array([ni])).ravel()
            q_restart[dofs] = restart_coords[ni]

        robot = robot.update(q=q_restart)
        print(f"  Restart at step {restart_step}, t = {restart_time:.6f} s")
        print(f"  Left-end y at restart: {restart_coords[0, 1]:.6f} m "
              f"(initial: {beam_y:.6f} m)")

    if restart:
        current_displacement = restart_coords[0, 1] - beam_y
        print(f"  Accumulated displacement at restart: {current_displacement:.6f} m")
    else:
        current_displacement = 0.0

    def before_step_fn(rb, t):
        """Identical kinematic clamp-velocity control to contact_bending_test.py."""
        nonlocal current_displacement

        dz_applied = 0.0
        if current_displacement < max_displacement:
            dt_local = rb.sim_params.dt
            dz_applied = min(imposed_velocity * dt_local, max_displacement - current_displacement)
            current_displacement += dz_applied

        if current_displacement >= max_displacement - 1e-9:
            rb.sim_params.total_time = min(rb.sim_params.total_time, t + rb.sim_params.dt)

        if dz_applied != 0.0:
            rb = rb.move_nodes(left_end_nodes, np.array([0.0, dz_applied, 0.0]))

        return rb

    stepper.before_step = before_step_fn

    def write_snapshot(rb, t_now, fh_coords, fh_forces):
        """Write one saved snapshot (beam coordinates + beam contact forces).

        Contact forces are queried directly from the time stepper's own
        IMCEnergy instance (the same object 'selfContact' built), at the
        state just reached. Using first_iter=False re-uses the broad-phase
        candidate list and adaptive k_c already established by the real
        Newton iterations of this step, without perturbing either (a
        first_iter=True call would recompute/overwrite both as a side
        effect, which must not happen from an external logging call).
        """
        q = rb.state.q
        node_coords = q[:rb.end_node_dof_index].reshape(-1, 3)
        beam_coords = node_coords[:n_beam_nodes_total]
        for node_idx, xyz in enumerate(beam_coords):
            fh_coords.write(
                f"{t_now:.6f}\t{node_idx}\t{xyz[0]:.12e}\t{xyz[1]:.12e}\t{xyz[2]:.12e}\n"
            )

        try:
            Fs, _ = stepper._contact_energy.grad_hess_energy(
                rb.state, rb, None, first_iter=False)
            contact_force_nodes = Fs[:rb.end_node_dof_index].reshape(-1, 3)[:n_beam_nodes_total]
        except Exception as e:
            print(f"Warning: Could not extract contact forces at t={t_now:.6f}s: {e}")
            contact_force_nodes = np.zeros_like(beam_coords)

        for node_idx, fxyz in enumerate(contact_force_nodes):
            fh_forces.write(
                f"{t_now:.6f}\t{node_idx}\t{fxyz[0]:.12e}\t{fxyz[1]:.12e}\t{fxyz[2]:.12e}\n"
            )
        return contact_force_nodes

    print("\nRunning simulation...")
    n_steps = int(round(total_time / dt))
    if abs(n_steps * dt - total_time) > 1e-12:
        raise ValueError(
            f"total_time={total_time} is not an integer multiple of dt={dt}. "
            "Please adjust dt or total_time."
        )

    saved_count = 0
    final_forces = None
    final_robot = robot
    file_mode = "a" if restart else "w"

    with output_coords_txt.open(file_mode, encoding="utf-8") as fh_coords, \
         output_forces_txt.open(file_mode, encoding="utf-8") as fh_forces:

        if not restart:
            fh_coords.write("# time_s\tnode_id\tx_m\ty_m\tz_m\n")
            fh_forces.write("# time_s\tnode_id\tfx_N\tfy_N\tfz_N\n")

            final_forces = write_snapshot(final_robot, 0.0, fh_coords, fh_forces)
            saved_count += 1
            fh_coords.flush()
            fh_forces.flush()

        start_step = restart_step if restart else 0
        if restart:
            print(f"  Resuming from step {start_step} / {n_steps}")

        for i in range(start_step, n_steps):
            t_before = i * dt
            if stepper.before_step is not None:
                final_robot = stepper.before_step(final_robot, t_before)

            final_robot, _ = stepper.step(final_robot)
            t_after = (i + 1) * dt

            if (i + 1) % sim_params.log_step == 0:
                final_forces = write_snapshot(final_robot, t_after, fh_coords, fh_forces)
                saved_count += 1
                fh_coords.flush()
                fh_forces.flush()

            print("current_time: ", t_after)

    print(f"Simulation completed at t={n_steps * dt:.6f} s with {saved_count} saved snapshots")
    print(f"Saved coordinates to '{output_coords_txt}'")
    print(f"Saved contact forces to '{output_forces_txt}'")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total snapshots saved: {saved_count}")
    print(f"Time range: {0.0:.4f} s to {n_steps * dt:.4f} s")
    print(f"Final max displacement: {current_displacement:.6f} m")

    u_final = final_robot.state.u
    u_nodes_final = u_final[:final_robot.end_node_dof_index].reshape(-1, 3)
    max_vel_final = np.max(np.linalg.norm(u_nodes_final[:n_beam_nodes_total], axis=1))
    print(f"Final max velocity: {max_vel_final:.6f} m/s")

    final_forces = final_forces[beam_node_indices, :]
    max_contact_force = np.max(np.linalg.norm(final_forces, axis=1))
    total_vertical_force = np.sum(final_forces[:, 1])
    print(f"Final max contact force magnitude: {max_contact_force:.6e} N")
    print(f"Final total vertical contact force (sum Fy): {total_vertical_force:.6e} N")

    with output_summary_txt.open("w", encoding="utf-8") as fh_summary:
        fh_summary.write("THREE-POINT BENDING TEST SUMMARY (2-node / segment model)\n")
        fh_summary.write(f"n_beam_nodes = {n_beam_nodes}\n")
        fh_summary.write(f"n_circle_nodes = {n_circle_nodes}\n")
        fh_summary.write(f"dt = {dt}\n")
        fh_summary.write(f"total_time = {total_time}\n")
        fh_summary.write(f"log_step = {sim_params.log_step}\n")
        fh_summary.write(f"save_interval_effective = {effective_save_interval}\n")
        fh_summary.write(f"snapshots_saved = {saved_count}\n")
        fh_summary.write(f"final_time = {n_steps * dt:.6f}\n")
        fh_summary.write(f"final_max_displacement = {current_displacement:.12e}\n")
        fh_summary.write(f"final_max_velocity = {max_vel_final:.12e}\n")
        fh_summary.write(f"final_max_contact_force = {max_contact_force:.12e}\n")
        fh_summary.write(f"final_total_vertical_contact_force = {total_vertical_force:.12e}\n")
    print(f"Saved summary to '{output_summary_txt}'")

    return {
        'beam_node_indices': beam_node_indices,
        'left_end_nodes': left_end_nodes,
        'coords_txt': str(output_coords_txt),
        'forces_txt': str(output_forces_txt),
        'summary_txt': str(output_summary_txt),
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = run_contact_bending_test_2node(restart=restart_mode)
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
