#!/usr/bin/env python3
"""
Three-point bending test with contact — simple bead-distance variant.

Identical to contact_bending_test_3D.py except for the contact model:
- Eberly arc-to-arc beam-to-beam contact is REPLACED by a simple bead distance
  between each beam node and the circular obstacle:

      gap_i = ||X_i - C|| - circle_radius - beam_radius

  Each beam node is treated as a sphere of radius `beam_radius`; the obstacle
  is a sphere of radius `circle_radius` centred at `C`.
- The circle obstacle is therefore a fixed point + radius, NOT a beam.

All other simulation parameters (geometry, material, dt, kc, d_imc, damping,
clamping, imposed velocity) are unchanged.
"""

import argparse
from pathlib import Path
import numpy as np
import dismech
from dismech.contact import compute_circle_node_bead_contact_forces

# ============================================================================
# PARAMETERS
# ============================================================================
beam_length = 1.0  # m
beam_radius = 0.001  # m
n_beam_nodes = 50  # 50 nodes to discretize the beam

# Initial clearance between the closest bead and the obstacle surface.
# Picked > d_imc so the system starts outside the activation band.
INITIAL_BEAD_GAP = 0.001  # m

parser = argparse.ArgumentParser(
    description="Three-point bending test with bead-distance contact")
parser.add_argument("n_nodes", nargs="?", type=int, default=n_beam_nodes,
                    help="Number of beam nodes (default: %(default)s)")
parser.add_argument("--restart", action="store_true",
                    help="Restart from the last saved snapshot")
parser.add_argument("--total_time", type=float, default=1000.0,
                    help="Total simulation time [s] (default: %(default)s)")
cli_args = parser.parse_args()
n_beam_nodes = cli_args.n_nodes
restart_mode = cli_args.restart
total_time = cli_args.total_time

# Circle obstacle
circle_center = np.array([0.8, 0.0, 0.0])  # m
circle_radius = 0.2  # m

# Material
E = 210e8
nu = 0.3
density = 7850

# Quasi-static / inextensibility
k_inext = 1e3
m_quasi = 0.01

# Velocity control
imposed_velocity = 1.0e-3
max_displacement = 1.2

# Simulation
dt = 1.0e-2
save_interval = 0.01

# ============================================================================
# GEOMETRY
# ============================================================================

def create_beam_geometry(length, n_nodes, y_pos, kappa0=-1e-3):
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")

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


# ============================================================================
# RESTART
# ============================================================================

def load_last_snapshot(coords_file, n_nodes):
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
            f"Could not find a complete snapshot in '{coords_file}'.")

    print(f"  Loaded restart snapshot at t = {last_time:.6f} s")
    return last_time, coords


# ============================================================================
# CROSS SECTION
# ============================================================================

def calculate_cross_section_properties(r):
    A = np.pi * r**2
    I = np.pi * r**4 / 4
    J = np.pi * r**4 / 2
    return A, I, J


# ============================================================================
# SIMULATION
# ============================================================================

def run_contact_bending_test_bead(restart=False):
    print("=" * 70)
    print("THREE-POINT BENDING TEST WITH CONTACT (Bead distance)")
    if restart:
        print(">>> RESTART MODE <<<")
    print("=" * 70)

    # Bead radius decoupled from the rod radius: half the inter-node
    # spacing so adjacent beads are tangent in the undeformed beam.
    bead_radius = beam_length / (2.0 * (n_beam_nodes - 1))

    # Initial beam y-position chosen so the closest bead starts INITIAL_BEAD_GAP
    # above the obstacle (i.e. just outside the activation band).
    beam_y = circle_center[1] - circle_radius - bead_radius - INITIAL_BEAD_GAP
    print(f"Bead radius = L / (2(N-1)) = {bead_radius:.6f} m "
          f"(rod radius = {beam_radius:.6f} m)")
    print(f"Initial beam y = {beam_y:.6f} m "
          f"(closest bead clearance = {INITIAL_BEAD_GAP:.6f} m)")

    beam_geo = create_beam_geometry(beam_length, n_beam_nodes, beam_y)
    n_beam_nodes_total = beam_geo.nodes.shape[0]
    beam_node_indices = np.arange(n_beam_nodes_total)
    print(f"Created beam: {n_beam_nodes_total} nodes (uniform mesh; clamp nodes [0,1])")
    print(f"Obstacle: sphere at {circle_center.tolist()} radius {circle_radius} m "
          f"(no obstacle geometry added)")

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

    n_edges_beam_A = n_beam_nodes_total - 1
    sim_params = dismech.SimParams(
        static_sim=False,
        two_d_sim=False,
        use_mid_edge=False,
        use_line_search=False,
        show_floor=False,
        log_data=True,
        log_step=log_step,
        dt=dt,
        max_iter=100,
        total_time=total_time,
        plot_step=0,
        tol=1e-4,
        ftol=1e-4,
        dtol=1e-4,
        n_edges_beam_A=n_edges_beam_A,
    )

    env = dismech.Environment()
    env.add_force(
        'circleNodeBead',
        circle_center=circle_center,
        circle_radius=circle_radius,
        beam_radius=beam_radius,
        bead_radius=bead_radius,
        kc=1.0,
        d_imc=0.0008,
        n_nodes_beam=n_beam_nodes_total,
    )
    env.add_force('damping', eta=1.0)

    robot = dismech.SoftRobot(geom, material, beam_geo, sim_params, env)
    print("Created SoftRobot")

    left_end_nodes = np.array([0, 1])
    robot = robot.fix_nodes(left_end_nodes, fix_edges=False)
    print(f"Fixed left end nodes {left_end_nodes}")

    free_beam_nodes = np.arange(2, n_beam_nodes_total)
    robot = robot.fix_nodes(free_beam_nodes, axis=2, fix_edges=False)
    print(f"Fixed z-DOF for beam nodes {free_beam_nodes[0]}..{free_beam_nodes[-1]}")

    stepper = dismech.ImplicitEulerTimeStepper(robot)

    out_dir = Path(__file__).resolve().parent / "contact_output_files_3D_bead"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_coords_txt = out_dir / f"reg_beam_coords_N{n_beam_nodes}_bead.txt"
    output_forces_txt = out_dir / f"reg_contact_forces_N{n_beam_nodes}_bead.txt"
    output_summary_txt = out_dir / f"reg_summary_N{n_beam_nodes}_bead.txt"

    restart_time = 0.0
    restart_step = 0

    if restart:
        restart_time, restart_coords = load_last_snapshot(
            output_coords_txt, n_beam_nodes_total)
        restart_step = int(round(restart_time / dt))

        q_restart = np.copy(robot.state.q)
        for ni in range(n_beam_nodes_total):
            dofs = robot.map_node_to_dof(np.array([ni])).ravel()
            q_restart[dofs] = restart_coords[ni]

        robot = robot.update(q=q_restart)
        print(f"  Restart at step {restart_step}, t = {restart_time:.6f} s")

    if restart:
        current_displacement = restart_coords[0, 1] - beam_y
        print(f"  Accumulated displacement at restart: {current_displacement:.6f} m")
    else:
        current_displacement = 0.0

    def before_step_fn(rb, t):
        nonlocal current_displacement

        dz_applied = 0.0

        if current_displacement < max_displacement:
            dt_local = rb.sim_params.dt
            dz_applied = min(imposed_velocity * dt_local,
                             max_displacement - current_displacement)
            current_displacement += dz_applied

        if current_displacement >= max_displacement - 1e-9:
            rb.sim_params.total_time = min(rb.sim_params.total_time,
                                           t + rb.sim_params.dt)

        if dz_applied != 0.0:
            rb = rb.move_nodes(left_end_nodes, np.array([0.0, dz_applied, 0.0]))

        return rb

    stepper.before_step = before_step_fn

    def write_snapshot(rb, t_now, fh_coords, fh_forces):
        q = rb.state.q
        node_coords = q[:rb.end_node_dof_index].reshape(-1, 3)
        for node_idx, xyz in enumerate(node_coords):
            fh_coords.write(
                f"{t_now:.6f}\t{node_idx}\t{xyz[0]:.12e}\t{xyz[1]:.12e}\t{xyz[2]:.12e}\n"
            )

        try:
            k_c_save = getattr(rb.env, '_nodebead_kc', rb.env.imc_stiffness)
            F_reg, _, _ = compute_circle_node_bead_contact_forces(
                rb, q,
                rb.env.circle_center_3d,
                rb.env.circle_radius,
                rb.env.bead_radius,
                k_c=k_c_save,
                n_nodes_beam=n_beam_nodes_total,
                d_imc=0.0008,
            )
            contact_force_nodes = F_reg[:rb.end_node_dof_index].reshape(-1, 3)
        except Exception as e:
            print(f"Warning: Could not extract contact forces at t={t_now:.6f}s: {e}")
            contact_force_nodes = np.zeros_like(node_coords)

        for node_idx, fxyz in enumerate(contact_force_nodes):
            fh_forces.write(
                f"{t_now:.6f}\t{node_idx}\t{fxyz[0]:.12e}\t{fxyz[1]:.12e}\t{fxyz[2]:.12e}\n"
            )
        return contact_force_nodes

    print("\nRunning simulation...")
    n_steps = int(round(total_time / dt))
    if abs(n_steps * dt - total_time) > 1e-12:
        raise ValueError(
            f"total_time={total_time} is not an integer multiple of dt={dt}.")

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
                final_forces = write_snapshot(final_robot, t_after,
                                              fh_coords, fh_forces)
                saved_count += 1
                fh_coords.flush()
                fh_forces.flush()

            print("current_time: ", t_after)

    print(f"Simulation completed at t={n_steps * dt:.6f} s "
          f"with {saved_count} saved snapshots")
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
    max_vel_final = np.max(np.linalg.norm(u_nodes_final, axis=1))
    print(f"Final max velocity: {max_vel_final:.6f} m/s")

    final_forces = final_forces[beam_node_indices, :]
    max_contact_force = np.max(np.linalg.norm(final_forces, axis=1))
    total_vertical_force = np.sum(final_forces[:, 1])
    print(f"Final max contact force magnitude: {max_contact_force:.6e} N")
    print(f"Final total vertical contact force (sum Fy): {total_vertical_force:.6e} N")

    with output_summary_txt.open("w", encoding="utf-8") as fh_summary:
        fh_summary.write("THREE-POINT BENDING TEST SUMMARY (Bead distance)\n")
        fh_summary.write(f"n_beam_nodes = {n_beam_nodes}\n")
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
    results = run_contact_bending_test_bead(restart=restart_mode)
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
