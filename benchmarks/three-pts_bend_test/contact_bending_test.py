#!/usr/bin/env python3
"""
Three-point bending test with contact: one half simulation.
Beam contacts a fixed circle obstacle while two left end points are driven upward by a prescribed velocity.

Geometry:
- Beam: 1 m from x=0 to x=1, y=-0.3, radius 0.001 m
- Fixed circle: center at x=0.8, y=0, radius 0.2 m
- Two left end points are driven upward

Simulation:
- 2D simulation
- Saves: time, node coordinates, contact forces per node
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import dismech
from dismech.contact import compute_circle_four_nodes_imc_weighted_forces_3d_xy

# ============================================================================
# PARAMETERS
# ============================================================================
# Beam parameters
beam_length = 1.0  # m
beam_y = -0.202  # m
beam_radius = 0.001  # m
# NOTE:
#   `n_beam_nodes` controls the total number of nodes along the full beam
#   length. We later fix the left two nodes (0 and 1) to represent the clamp.
n_beam_nodes = 10  # default number of nodes along the beam

# CLI argument parsing
parser = argparse.ArgumentParser(description="Three-point bending test with contact")
parser.add_argument("n_nodes", nargs="?", type=int, default=n_beam_nodes,
                    help="Number of beam nodes (default: %(default)s)")
parser.add_argument("--restart", action="store_true",
                    help="Restart from the last saved snapshot in the coordinates file")
cli_args = parser.parse_args()
n_beam_nodes = cli_args.n_nodes
restart_mode = cli_args.restart
if n_beam_nodes != 10:
    print(f"Using n_beam_nodes = {n_beam_nodes} from command-line argument.")
if restart_mode:
    print(f"RESTART MODE enabled: will resume from last saved snapshot.")

# Circle obstacle parameters
circle_center = np.array([0.8, 0.0, 0.0])  # m
circle_radius = 0.2  # m
n_circle_nodes = 200  # Number of nodes to approximate circle

# Material properties
E = 210e8  # Young's modulus [Pa] = 210 GPa
nu = 0.3  # Poisson's ratio
density = 7850  # Steel density [kg/m³]

# Quasi-static / inextensibility scaling factors
# ------------------------------------------------
# k_inext > 1: increases axial stiffness EA (approximate inextensibility)
# m_quasi < 1: reduces effective mass per length for more quasi-static response
# Set both to 1.0 to recover the physical, unscaled case.
k_inext = 1e3   # inextensibility factor (axial stiffness scaling)
m_quasi = 0.01   # quasi-static factor (mass per length scaling, < 1)

# Velocity control parameters
#  - Clamp driver advances by dz = imposed_velocity * dt (upward along +y)
#    at every time step until `max_displacement` is reached.
#  - Set `imposed_velocity` so that reaching `max_displacement` = 1.2 m occurs
#    within 1200 s of simulation time: imposed_velocity = 1.2 / 1200 = 1e-3 m/s.
imposed_velocity = 1.0e-3  # [m/s]
max_displacement = 1.2  # Maximum upward displacement [m]

# Simulation parameters
dt = 1.0e-3  # Time step [s]
total_time = 1000.0  # Total simulation time [s]
# Store results every `save_interval` seconds to reduce memory usage.
#  - 0.01 s  -> keep 1 in 2 steps (with dt = 0.005 s)
#  - 0.10 s  -> keep 1 in 20 steps
save_interval = 0.01  # [s]

# ============================================================================
# GEOMETRY CREATION
# ============================================================================

def create_beam_geometry(length, n_nodes, y_pos, kappa0=-1e-3):
    """
    Create a beam with constant downward curvature kappa0, starting from a
    straight left-end edge (nodes 0-1 are horizontal at y=y_pos).

    The arc is parameterised by arc-length s from the left end.
    Tangent angle: theta(s) = kappa0 * s
    Position:      x(s) = integral cos(theta), y(s) = y_pos + integral sin(theta)

    With |kappa0| ~ 1e-6 the beam looks straight but the discrete curvature
    is always the same sign, avoiding the kappa=0 discontinuity.
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2 to create a 1D beam mesh")

    # Arc-length stations (uniform)
    s = np.linspace(0.0, length, n_nodes)

    # Tangent angle at each station (theta=0 at s=0 → left edge is horizontal)
    theta = kappa0 * s

    # Integrate to get positions (cumulative trapezoidal)
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
    """Create a circle geometry approximated by a polygon."""
    nodes = np.zeros((n_nodes, 3))
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    
    nodes[:, 0] = center[0] + radius * np.cos(angles)
    nodes[:, 1] = center[1] + radius * np.sin(angles)
    nodes[:, 2] = center[2]
    
    # Create edges connecting consecutive nodes (closed loop)
    edges = np.array([[i, (i+1) % n_nodes] for i in range(n_nodes)], dtype=int)
    face_nodes = np.empty((0, 3), dtype=int)
    
    return dismech.Geometry(nodes, edges, face_nodes)


def combine_geometries(geo1, geo2):
    """Combine two geometries into one."""
    n1 = geo1.nodes.shape[0]
    n2 = geo2.nodes.shape[0]
    
    # Combine nodes
    combined_nodes = np.vstack([geo1.nodes, geo2.nodes])
    
    # Offset edge indices for second geometry
    edges2_offset = geo2.edges + n1
    combined_edges = np.vstack([geo1.edges, edges2_offset])
    
    # Combine faces (if any)
    if geo1.face_nodes.size > 0 and geo2.face_nodes.size > 0:
        face_nodes2_offset = geo2.face_nodes + n1
        combined_faces = np.vstack([geo1.face_nodes, face_nodes2_offset])
    elif geo1.face_nodes.size > 0:
        combined_faces = geo1.face_nodes
    elif geo2.face_nodes.size > 0:
        combined_faces = geo2.face_nodes + n1
    else:
        combined_faces = np.empty((0, 3), dtype=int)
    
    return dismech.Geometry(combined_nodes, combined_edges, combined_faces)


# ============================================================================
# RESTART UTILITIES
# ============================================================================

def load_last_snapshot(coords_file, n_nodes):
    """
    Read the last complete snapshot from a coordinates file.

    Returns
    -------
    restart_time : float
        Simulation time of the last snapshot.
    node_coords : np.ndarray, shape (n_nodes, 3)
        Node coordinates [x, y, z] for each node.
    """
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
                # New snapshot started
                if last_time is not None and lines_in_snapshot < n_nodes:
                    # Previous snapshot was incomplete — skip it
                    pass
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
    """Calculate cross-section properties for circular beam."""
    A = np.pi * r**2  # Area
    I = np.pi * r**4 / 4  # Moment of inertia
    J = np.pi * r**4 / 2  # Torsional constant
    return A, I, J


# ============================================================================
# SIMULATION
# ============================================================================

def run_contact_bending_test(restart=False):
    """Run the contact bending test."""
    print("="*70)
    print("THREE-POINT BENDING TEST WITH CONTACT")
    if restart:
        print(">>> RESTART MODE <<<")
    print("="*70)
    
    # Create beam geometry with a uniform mesh; clamp is enforced by fixing nodes later.
    beam_geo = create_beam_geometry(beam_length, n_beam_nodes, beam_y)
    n_beam_nodes_total = beam_geo.nodes.shape[0]
    beam_node_indices = np.arange(n_beam_nodes_total)
    print(f"✓ Created beam geometry: {n_beam_nodes_total} nodes "
          f"(uniform mesh; fixed clamp nodes [0, 1])")
    
    # Calculate cross-section properties
    A, I, J = calculate_cross_section_properties(beam_radius)

    # ----------------------------------------------------------------------
    # Inextensible + quasi-static approximation
    # ----------------------------------------------------------------------
    # We want:
    #   - Very stiff in extension (EA large)  → approximate inextensibility
    #   - Same bending stiffness EI
    #   - Reduced inertia (smaller rho*A)    → more quasi-static response
    #
    # Strategy:
    #   1) Multiply area by k_inext to increase EA.
    #   2) Keep I unchanged so EI stays the same.
    #   3) Divide density by k_inext to keep lineic mass unchanged.
    #   4) Multiply density by m_quasi < 1 to reduce dynamics further.
    #
    # NOTE:
    #   k_inext and m_quasi are defined at module level so they can be
    #   overridden externally (e.g. from run_contact_bending_study.py).
    #   Setting both to 1.0 recovers the physical, unscaled case.

    A_eff = k_inext * A   # large EA
    I_eff = I             # same EI
    J_eff = J

    # First keep rho*A the same, then scale down by m_quasi
    density_eff = (density / k_inext) * m_quasi

    # Geometry parameters (mechanical section properties)
    geom = dismech.GeomParams(
        rod_r0=beam_radius,  # keep geometric radius for contact
        shell_h=0,
        axs=A_eff,
        ixs1=I_eff,
        ixs2=I_eff,
        jxs=J_eff
    )
    
    # Material properties (with modified density for quasi-static response)
    material = dismech.Material(
        density=density_eff,
        youngs_rod=E,
        youngs_shell=0,
        poisson_rod=nu,
        poisson_shell=0
    )
    
    # Compute logging decimation from desired save interval.
    # This directly reduces memory usage because fewer robot states are retained.
    log_step = max(1, int(round(save_interval / dt)))
    effective_save_interval = log_step * dt
    print(f"✓ Saving state every {effective_save_interval:.4f} s (log_step={log_step})")

    # Simulation parameters (2D, dynamic with quasi-static control)
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
        plot_step=0,  # Disable internal plotting
        tol=1e-5,
        ftol=1e-5,
        dtol=1e-5
    )
    
    # Environment with regularized circle contact and damping
    env = dismech.Environment()
    env.add_force(
        'circleFourNodesIMCWeightedXY',
        circle_center=circle_center,
        circle_radius=circle_radius,
        beam_radius=beam_radius,
        kc=1.0,
    )
    env.add_force('damping', eta=1.0)  # Damping for quasi-static
    
    # Create robot (beam only; circle is analytic, not discretized here)
    robot = dismech.SoftRobot(geom, material, beam_geo, sim_params, env)
    print("✓ Created SoftRobot")
    
    # Identify the two left end points of the beam (nodes 0 and 1)
    left_end_nodes = np.array([0, 1])
    
    # Fix the two left end nodes so we can control their positions directly
    # This makes them fully controlled - we set their positions explicitly
    robot = robot.fix_nodes(left_end_nodes, fix_edges=False)
    print(f"✓ Fixed left end nodes {left_end_nodes} for position control")
    
    # Create time stepper
    stepper = dismech.ImplicitEulerTimeStepper(robot)
    
    # Progressive output paths (stream to disk to keep memory low)
    out_dir = Path(__file__).resolve().parent / "contact_output_files"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_coords_txt = out_dir / f"reg_beam_coords_N{n_beam_nodes}.txt"
    output_forces_txt = out_dir / f"reg_contact_forces_N{n_beam_nodes}.txt"
    output_summary_txt = out_dir / f"reg_summary_N{n_beam_nodes}.txt"

    # ------------------------------------------------------------------
    # Restart: load last snapshot and reconstruct state
    # ------------------------------------------------------------------
    restart_time = 0.0
    restart_step = 0

    if restart:
        restart_time, restart_coords = load_last_snapshot(
            output_coords_txt, n_beam_nodes_total
        )
        restart_step = int(round(restart_time / dt))

        # Build the q vector from saved node coordinates
        q_restart = np.copy(robot.state.q)
        for ni in range(n_beam_nodes_total):
            dofs = robot.map_node_to_dof(np.array([ni])).ravel()
            q_restart[dofs] = restart_coords[ni]

        # Set q on the robot (u = 0 is fine for quasi-static)
        robot = robot.update(q=q_restart)

        # Also need to update the fixed-node positions in the robot so they
        # match the restart coordinates (otherwise fix_nodes constraints
        # would pull them back to the initial geometry).
        # move_nodes_to is not available, so we use move_nodes with delta=0
        # Actually the fixed DOFs in q are already set above via update(q=...).
        # The boundary conditions just prevent the solver from changing them.

        print(f"  Restart at step {restart_step}, t = {restart_time:.6f} s")
        print(f"  Left-end y at restart: {restart_coords[0, 1]:.6f} m "
              f"(initial: {beam_y:.6f} m)")

    # Track current displacement (used for stopping and logging).
    # On restart, recompute from the actual node position vs initial position.
    if restart:
        current_displacement = restart_coords[0, 1] - beam_y
        print(f"  Accumulated displacement at restart: {current_displacement:.6f} m")
    else:
        current_displacement = 0.0

    def before_step_fn(rb, t):
        """Apply imposed velocity (as a kinematic clamp motion) each step."""
        nonlocal current_displacement

        dz_applied = 0.0

        # ------------------------------------------------------------------
        # Velocity control:
        #  - Primary goal: reach max_displacement
        #  - Safety: stop if total_time is reached
        # ------------------------------------------------------------------

        # Always advance the clamp by imposed_velocity * dt until we hit max_displacement.
        if current_displacement < max_displacement:
            dt_local = rb.sim_params.dt
            dz_applied = min(imposed_velocity * dt_local, max_displacement - current_displacement)
            current_displacement += dz_applied

        # If we've effectively reached the target displacement, we can shorten
        # the simulation by updating total_time so that the time-stepper loop
        # stops shortly after this point (safety limit remains as configured).
        if current_displacement >= max_displacement - 1e-9:
            # Clamp total_time to the current time (or very slightly beyond)
            rb.sim_params.total_time = min(rb.sim_params.total_time, t + rb.sim_params.dt)

        # Apply kinematic clamp motion (relative move) if we advanced in displacement.
        # Left end nodes are fixed DOFs, so we update their constrained positions here.
        if dz_applied != 0.0:
            rb = rb.move_nodes(left_end_nodes, np.array([0.0, dz_applied, 0.0]))

        return rb

    stepper.before_step = before_step_fn

    def write_snapshot(rb, t_now, fh_coords, fh_forces):
        """Write one saved snapshot (coordinates + contact forces) to text files."""
        q = rb.state.q
        node_coords = q[:rb.end_node_dof_index].reshape(-1, 3)
        for node_idx, xyz in enumerate(node_coords):
            fh_coords.write(
                f"{t_now:.6f}\t{node_idx}\t{xyz[0]:.12e}\t{xyz[1]:.12e}\t{xyz[2]:.12e}\n"
            )

        try:
            F_reg, _, _, _ = compute_circle_four_nodes_imc_weighted_forces_3d_xy(
                rb,
                q,
                circle_center,
                circle_radius,
                beam_radius,
            )
            contact_force_nodes = F_reg[:rb.end_node_dof_index].reshape(-1, 3)
        except Exception as e:
            print(f"Warning: Could not extract regularized contact forces at t={t_now:.6f}s: {e}")
            contact_force_nodes = np.zeros_like(node_coords)

        for node_idx, fxyz in enumerate(contact_force_nodes):
            fh_forces.write(
                f"{t_now:.6f}\t{node_idx}\t{fxyz[0]:.12e}\t{fxyz[1]:.12e}\t{fxyz[2]:.12e}\n"
            )
        return contact_force_nodes

    # Run simulation with explicit fixed step count to end at total_time exactly.
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

    # On restart: append to existing files; on fresh start: write new files
    file_mode = "a" if restart else "w"

    with output_coords_txt.open(file_mode, encoding="utf-8") as fh_coords, \
         output_forces_txt.open(file_mode, encoding="utf-8") as fh_forces:

        if not restart:
            fh_coords.write("# time_s\tnode_id\tx_m\ty_m\tz_m\n")
            fh_forces.write("# time_s\tnode_id\tfx_N\tfy_N\tfz_N\n")

            # Save initial state at t=0
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

    print(f"✓ Simulation completed at t={n_steps * dt:.6f} s with {saved_count} saved snapshots")
    print(f"✓ Saved coordinates progressively to '{output_coords_txt}'")
    print(f"✓ Saved contact forces progressively to '{output_forces_txt}'")
    
    # # Also (optionally) save as .mat file for MATLAB compatibility
    # # NOTE: For very long simulations this file can be huge. We therefore
    # # downsample in time before writing, and if SciPy/Matlab format cannot
    # # handle it we just skip the .mat without failing the script.
    # try:
    #     import scipy.io
    #     output_file_mat = "contact_bending_test_results.mat"

    #     # Downsample factor for .mat export (keeps every N-th time step)
    #     mat_downsample = 100  # adjust if needed
    #     idx = np.arange(0, time_history.shape[0], mat_downsample)

    #     time_mat = time_history[idx]
    #     node_coords_mat = node_coords_history[idx]
    #     contact_forces_mat = contact_forces_history[idx]

    #     try:
    #         scipy.io.savemat(
    #             output_file_mat,
    #             {
    #                 'time': time_mat,
    #                 'node_coordinates': node_coords_mat,
    #                 'contact_forces': contact_forces_mat,
    #                 'beam_node_indices': beam_node_indices,
    #                 'circle_node_indices': circle_node_indices,
    #                 'left_end_nodes': left_end_nodes
    #             }
    #         )
    #         print(f"✓ Saved downsampled results to '{output_file_mat}' (MATLAB format)")
    #         print(f"  (Saved every {mat_downsample}-th time step; full data is in the .npz file.)")
    #     except Exception as e:
    #         print("  Warning: could not save .mat file (likely too large for Matlab v5 format).")
    #         print(f"  Error was: {e}")
    #         print("  You can work directly from 'contact_bending_test_results.npz' in MATLAB using load_npz.m.")
    # except ImportError:
    #     print("  Note: scipy not available, .mat file not created.")
    #     print("  Install scipy or run convert_npz_to_mat.py to convert .npz to .mat")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total snapshots saved: {saved_count}")
    print(f"Time range: {0.0:.4f} s to {n_steps * dt:.4f} s")
    print(f"Final max displacement: {current_displacement:.6f} m")
    
    # Check final velocities
    u_final = final_robot.state.u
    u_nodes_final = u_final[:final_robot.end_node_dof_index].reshape(-1, 3)
    max_vel_final = np.max(np.linalg.norm(u_nodes_final, axis=1))
    print(f"Final max velocity: {max_vel_final:.6f} m/s")
    
    # Check contact forces
    final_forces = final_forces[beam_node_indices, :]
    max_contact_force = np.max(np.linalg.norm(final_forces, axis=1))
    total_vertical_force = np.sum(final_forces[:, 1])  # Sum of Fy components
    print(f"Final max contact force magnitude: {max_contact_force:.6e} N")
    print(f"Final total vertical contact force (sum Fy): {total_vertical_force:.6e} N")
    print(f"  (Note: negative means circle pushes down on beam, positive means up)")

    with output_summary_txt.open("w", encoding="utf-8") as fh_summary:
        fh_summary.write("THREE-POINT BENDING TEST SUMMARY\n")
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
    print(f"✓ Saved summary to '{output_summary_txt}'")
    
    return {
        'time': None,
        'node_coordinates': None,
        'contact_forces': None,
        'beam_node_indices': beam_node_indices,
        'left_end_nodes': left_end_nodes,
        'robots': None,
        'coords_txt': str(output_coords_txt),
        'forces_txt': str(output_forces_txt),
        'summary_txt': str(output_summary_txt),
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = run_contact_bending_test(restart=restart_mode)
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
