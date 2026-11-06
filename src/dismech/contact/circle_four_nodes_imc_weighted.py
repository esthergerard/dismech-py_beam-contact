import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import math
from typing import Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:  # Avoid circular import at runtime
    from ..soft_robot import SoftRobot


# =============================================================================
# NOTE:
# The functions below implement the circular-arc IMC contact geometry and
# energy law in 2D. The last helper, `compute_circle_four_nodes_imc_weighted_forces_3d`,
# is a thin wrapper that maps Dismech's 3D DOFs onto that 2D model.
# =============================================================================


def computeE_IMC(delta_c, beam_radius, d_imc, K_imc):
    """
    Computes the E_imc value based on input parameters using NumPy.
    """
    d_imc = np.asarray(d_imc)
    K_imc = np.asarray(K_imc)
    delta_c = np.asarray(delta_c)
    body_R = np.asarray(beam_radius)

    term_R_minus_delta = body_R - delta_c
    threshold1 = body_R - d_imc
    threshold2 = body_R + d_imc

    E_imc = np.zeros_like(d_imc + K_imc + delta_c + body_R, dtype=float)

    mask1 = delta_c <= threshold1
    val_cond1 = term_R_minus_delta**2
    E_imc = np.where(mask1, val_cond1, E_imc)

    mask2 = (delta_c > threshold1) & (delta_c <= threshold2)
    log_arg = 1 + np.exp(K_imc * term_R_minus_delta)
    with np.errstate(divide='ignore', invalid='ignore'):
        val_cond2 = ((1 / K_imc) * np.log(log_arg)) ** 2
    E_imc = np.where(mask2, val_cond2, E_imc)

    return E_imc


def compute_dE_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc):
    """
    First derivative of E_imc with respect to delta_c.
    """
    delta_c = np.asarray(delta_c)
    beam_radius = np.asarray(beam_radius)
    d_imc = np.asarray(d_imc)
    K_imc = np.asarray(K_imc)

    term_R_minus_delta = beam_radius - delta_c
    threshold1 = beam_radius - d_imc
    threshold2 = beam_radius + d_imc

    dE_imc_ddelta_c = np.zeros_like(delta_c, dtype=float)

    mask1 = delta_c <= threshold1
    val_cond1 = -2 * term_R_minus_delta
    dE_imc_ddelta_c = np.where(mask1, val_cond1, dE_imc_ddelta_c)

    mask2 = (delta_c > threshold1) & (delta_c <= threshold2)
    log_arg = 1 + np.exp(K_imc * term_R_minus_delta)
    with np.errstate(divide='ignore', invalid='ignore'):
        val_cond2 = (-2 / K_imc) * (np.exp(K_imc * term_R_minus_delta) / log_arg) * np.log(log_arg)
    dE_imc_ddelta_c = np.where(mask2, val_cond2, dE_imc_ddelta_c)

    return dE_imc_ddelta_c


def compute_d2E_IMC_ddelta_c(delta_c, beam_radius, d_imc, K_imc):
    """
    Second derivative of E_imc with respect to delta_c.
    """
    delta_c = np.asarray(delta_c)
    beam_radius = np.asarray(beam_radius)
    d_imc = np.asarray(d_imc)
    K_imc = np.asarray(K_imc)

    term_R_minus_delta = beam_radius - delta_c
    threshold1 = beam_radius - d_imc
    threshold2 = beam_radius + d_imc

    d2E_imc_ddelta_c = np.zeros_like(delta_c, dtype=float)

    mask1 = delta_c <= threshold1
    d2E_imc_ddelta_c = np.where(mask1, 2, d2E_imc_ddelta_c)

    mask2 = (delta_c > threshold1) & (delta_c <= threshold2)
    v = np.exp(K_imc * term_R_minus_delta)
    u = 1 + v
    with np.errstate(divide='ignore', invalid='ignore'):
        val_cond2 = 2 * v / u ** 2 * (v + np.log(u))
    d2E_imc_ddelta_c = np.where(mask2, val_cond2, d2E_imc_ddelta_c)

    return d2E_imc_ddelta_c


def compute_curvature_numpy(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p2
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if mag1 < 1e-12 or mag2 < 1e-12:
        return 0.0
    kappa = 2 * cross / (mag1 * mag2 * (mag1 + mag2))
    return kappa


def compute_curvature_jax(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p2
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    mag1 = jnp.linalg.norm(v1)
    mag2 = jnp.linalg.norm(v2)
    mag1 = jnp.maximum(mag1, 1e-12)
    mag2 = jnp.maximum(mag2, 1e-12)
    kappa = 2 * cross / (mag1 * mag2 * (mag1 + mag2))
    return kappa


def fix_arc_angles_numpy(angle_mid1, angle_mid2, kappa, C_beam, mid1, mid2, R_curv):
    angle_mid1 = angle_mid1 % (2 * np.pi)
    angle_mid2 = angle_mid2 % (2 * np.pi)
    if kappa > 0:
        if angle_mid2 < angle_mid1:
            angle_mid2 += 2 * np.pi
    else:
        if angle_mid1 < angle_mid2:
            angle_mid1 += 2 * np.pi
    return angle_mid1, angle_mid2


def fix_arc_angles_jax(angle_mid1, angle_mid2, kappa, C_beam, mid1, mid2, R_curv):
    angle_mid1 = angle_mid1 % (2 * jnp.pi)
    angle_mid2 = angle_mid2 % (2 * jnp.pi)
    angle_mid2 = jnp.where(
        (kappa > 0) & (angle_mid2 < angle_mid1),
        angle_mid2 + 2 * jnp.pi,
        angle_mid2,
    )
    angle_mid1 = jnp.where(
        (kappa < 0) & (angle_mid1 < angle_mid2),
        angle_mid1 + 2 * jnp.pi,
        angle_mid1,
    )
    return angle_mid1, angle_mid2


def compute_extended_arc_gap_numpy(p1, p2, p3, circle_center, circle_radius, beam_radius):
    kappa = compute_curvature_numpy(p1, p2, p3)
    if abs(kappa) < 1e-8:
        kappa = 1e-8 if kappa >= 0 else -1e-8
    R_beam = 1.0 / kappa
    R_curv = abs(R_beam)

    mid1 = (p1 + p2) / 2
    mid2 = (p2 + p3) / 2

    chord_midpoint = (mid1 + mid2) / 2
    chord_vector = mid2 - mid1
    chord_length = np.linalg.norm(chord_vector)
    if chord_length <= 1e-12:
        return 1e6, 0.0
    if abs(R_beam) < chord_length / 2:
        return 1e6, 0.0

    perp_direction = np.array([-chord_vector[1], chord_vector[0]]) / chord_length
    L_half = chord_length / 2
    d = np.sqrt(R_beam**2 - L_half**2)

    if kappa > 0:
        C_beam = chord_midpoint + d * perp_direction
    else:
        C_beam = chord_midpoint - d * perp_direction

    angle_mid1 = math.atan2(mid1[1] - C_beam[1], mid1[0] - C_beam[0])
    angle_mid2 = math.atan2(mid2[1] - C_beam[1], mid2[0] - C_beam[0])
    angle_mid1, angle_mid2 = fix_arc_angles_numpy(
        angle_mid1, angle_mid2, kappa, C_beam, mid1, mid2, R_curv
    )

    if kappa > 0:
        original_angle_span = angle_mid2 - angle_mid1
    else:
        original_angle_span = angle_mid1 - angle_mid2
    if original_angle_span < 0:
        original_angle_span += 2 * np.pi

    arc_extension = original_angle_span / 2
    if kappa > 0:
        angle_mid1_ext = angle_mid1 - arc_extension
        angle_mid2_ext = angle_mid2 + arc_extension
    else:
        angle_mid1_ext = angle_mid1 + arc_extension
        angle_mid2_ext = angle_mid2 - arc_extension

    direction_to_circle = circle_center - C_beam
    direction_length = np.linalg.norm(direction_to_circle)
    if direction_length <= 1e-12:
        gap = abs(R_curv - circle_radius - beam_radius)
        return gap, 0.0

    angle_to_circle = math.atan2(direction_to_circle[1], direction_to_circle[0])

    def angle_in_range(angle, start, end):
        angle = angle % (2 * np.pi)
        start = start % (2 * np.pi)
        end = end % (2 * np.pi)
        if start <= end:
            return start <= angle <= end
        else:
            return angle >= start or angle <= end

    if kappa > 0:
        in_range = angle_in_range(angle_to_circle, angle_mid1_ext, angle_mid2_ext)
    else:
        in_range = angle_in_range(angle_to_circle, angle_mid2_ext, angle_mid1_ext)

    if in_range:
        closest_angle = angle_to_circle
    else:
        def angular_distance(a1, a2):
            diff = (a2 - a1) % (2 * np.pi)
            return min(diff, 2 * np.pi - diff)

        dist_to_start = angular_distance(angle_to_circle, angle_mid1_ext)
        dist_to_end = angular_distance(angle_to_circle, angle_mid2_ext)
        closest_angle = angle_mid1_ext if dist_to_start < dist_to_end else angle_mid2_ext

    if kappa > 0:
        total_arc_span = angle_mid2_ext - angle_mid1_ext
        if total_arc_span < 0:
            total_arc_span += 2 * np.pi
        angle_diff = closest_angle - angle_mid1_ext
        if angle_diff < 0:
            angle_diff += 2 * np.pi
        s_i_normalized = angle_diff / total_arc_span if total_arc_span > 0 else 0
    else:
        total_arc_span = angle_mid1_ext - angle_mid2_ext
        if total_arc_span < 0:
            total_arc_span += 2 * np.pi
        angle_diff = angle_mid1_ext - closest_angle
        if angle_diff < 0:
            angle_diff += 2 * np.pi
        s_i_normalized = angle_diff / total_arc_span if total_arc_span > 0 else 0

    s_i = 2 * s_i_normalized - 1

    closest_point = C_beam + R_curv * np.array(
        [np.cos(closest_angle), np.sin(closest_angle)]
    )
    distance_to_circle = np.linalg.norm(closest_point - circle_center)
    gap = distance_to_circle - circle_radius - beam_radius

    return gap, s_i


def compute_extended_arc_gap_jax(p1, p2, p3, circle_center, circle_radius, beam_radius):
    kappa = compute_curvature_jax(p1, p2, p3)
    kappa = jnp.where(
        jnp.abs(kappa) < 1e-8,
        jnp.where(kappa >= 0, 1e-8, -1e-8),
        kappa,
    )

    R_beam = 1.0 / kappa
    R_curv = jnp.abs(R_beam)

    mid1 = (p1 + p2) / 2
    mid2 = (p2 + p3) / 2

    chord_midpoint = (mid1 + mid2) / 2
    chord_vector = mid2 - mid1
    chord_length = jnp.linalg.norm(chord_vector)
    chord_length = jnp.maximum(chord_length, 1e-12)

    valid_radius = jnp.abs(R_beam) >= chord_length / 2

    perp_direction = jnp.array([-chord_vector[1], chord_vector[0]]) / chord_length

    L_half = chord_length / 2
    d_squared = R_beam**2 - L_half**2
    d_squared = jnp.maximum(d_squared, 1e-12)
    d = jnp.sqrt(d_squared)

    C_beam = jnp.where(
        kappa > 0,
        chord_midpoint + d * perp_direction,
        chord_midpoint - d * perp_direction,
    )

    angle_mid1 = jnp.arctan2(mid1[1] - C_beam[1], mid1[0] - C_beam[0])
    angle_mid2 = jnp.arctan2(mid2[1] - C_beam[1], mid2[0] - C_beam[0])
    angle_mid1, angle_mid2 = fix_arc_angles_jax(
        angle_mid1, angle_mid2, kappa, C_beam, mid1, mid2, R_curv
    )

    original_angle_span = jnp.where(
        kappa > 0,
        angle_mid2 - angle_mid1,
        angle_mid1 - angle_mid2,
    )
    original_angle_span = jnp.where(
        original_angle_span < 0,
        original_angle_span + 2 * jnp.pi,
        original_angle_span,
    )

    arc_extension = original_angle_span / 2
    angle_mid1_ext = jnp.where(
        kappa > 0,
        angle_mid1 - arc_extension,
        angle_mid1 + arc_extension,
    )
    angle_mid2_ext = jnp.where(
        kappa > 0,
        angle_mid2 + arc_extension,
        angle_mid2 - arc_extension,
    )

    direction_to_circle = circle_center - C_beam
    direction_length = jnp.linalg.norm(direction_to_circle)
    direction_length = jnp.maximum(direction_length, 1e-12)

    angle_to_circle = jnp.arctan2(direction_to_circle[1], direction_to_circle[0])

    angle_to_circle_norm = angle_to_circle % (2 * jnp.pi)
    angle_start_norm = angle_mid1_ext % (2 * jnp.pi)
    angle_end_norm = angle_mid2_ext % (2 * jnp.pi)

    in_range_pos = jnp.where(
        angle_start_norm <= angle_end_norm,
        (angle_to_circle_norm >= angle_start_norm)
        & (angle_to_circle_norm <= angle_end_norm),
        (angle_to_circle_norm >= angle_start_norm)
        | (angle_to_circle_norm <= angle_end_norm),
    )

    in_range_neg = jnp.where(
        angle_end_norm <= angle_start_norm,
        (angle_to_circle_norm >= angle_end_norm)
        & (angle_to_circle_norm <= angle_start_norm),
        (angle_to_circle_norm >= angle_end_norm)
        | (angle_to_circle_norm <= angle_start_norm),
    )

    in_range = jnp.where(kappa > 0, in_range_pos, in_range_neg)

    def angular_distance_jax(a1, a2):
        diff = (a2 - a1) % (2 * jnp.pi)
        return jnp.minimum(diff, 2 * jnp.pi - diff)

    dist_to_start = angular_distance_jax(angle_to_circle, angle_mid1_ext)
    dist_to_end = angular_distance_jax(angle_to_circle, angle_mid2_ext)

    closest_angle = jnp.where(
        in_range,
        angle_to_circle,
        jnp.where(dist_to_start < dist_to_end, angle_mid1_ext, angle_mid2_ext),
    )

    closest_point = C_beam + R_curv * jnp.array(
        [jnp.cos(closest_angle), jnp.sin(closest_angle)]
    )
    distance_to_circle = jnp.linalg.norm(closest_point - circle_center)
    gap = distance_to_circle - circle_radius - beam_radius

    gap = jnp.where(valid_radius, gap, 1e6)
    return gap


def compute_gap_with_si(p1, p2, p3, circle_center, circle_radius, beam_radius):
    return compute_extended_arc_gap_numpy(p1, p2, p3, circle_center, circle_radius, beam_radius)


def create_gap_functions(circle_center, circle_radius, beam_radius):
    def gap_function(points, circle_center_arg, circle_radius_arg, beam_radius_arg):
        p1 = points[:2]
        p2 = points[2:4]
        p3 = points[4:6]
        return compute_extended_arc_gap_jax(
            p1, p2, p3, circle_center_arg, circle_radius_arg, beam_radius_arg
        )

    def gap_function_frozen(points):
        return gap_function(points, circle_center, circle_radius, beam_radius)

    grad_function = jax.grad(gap_function_frozen)
    hessian_function = jax.hessian(gap_function_frozen)
    grad_function = jax.jit(grad_function)
    hessian_function = jax.jit(hessian_function)
    return gap_function_frozen, grad_function, hessian_function


# Cache for JIT-compiled gap gradient/Hessian (weighted contact)
_gap_functions_cache_weighted = {}


def _get_gap_functions_cached(circle_center, circle_radius, beam_radius):
    """
    Return cached JIT-compiled gradient and Hessian functions for the given
    circle parameters, creating and warming them once if needed.
    """
    circle_key = tuple(np.asarray(circle_center, dtype=float).tolist())
    cache_key = (circle_key, float(circle_radius), float(beam_radius))

    if cache_key not in _gap_functions_cache_weighted:
        circle_center_jax = jnp.array(circle_center)
        _, grad_fn_jit, hess_fn_jit = create_gap_functions(
            circle_center_jax, circle_radius, beam_radius
        )
        # Warm up compilation with a dummy call
        dummy_points = jnp.zeros(6)
        _ = grad_fn_jit(dummy_points)
        _ = hess_fn_jit(dummy_points)
        _gap_functions_cache_weighted[cache_key] = (grad_fn_jit, hess_fn_jit)

    return _gap_functions_cache_weighted[cache_key]


def compute_gap_jax(p1, p2, p3, circle_center, circle_radius, beam_radius):
    return compute_extended_arc_gap_jax(p1, p2, p3, circle_center, circle_radius, beam_radius)


def compute_gap_gradient(p1, p2, p3, circle_center, circle_radius, beam_radius):
    points = jnp.concatenate([p1, p2, p3])
    _, grad_fn, _ = create_gap_functions(circle_center, circle_radius, beam_radius)
    return grad_fn(points)


def compute_gap_hessian(p1, p2, p3, circle_center, circle_radius, beam_radius):
    points = jnp.concatenate([p1, p2, p3])
    _, _, hessian_fn = create_gap_functions(circle_center, circle_radius, beam_radius)
    return hessian_fn(points)


def compute_weights_from_s(s):
    k = 0.8204
    weight = 1 + (np.arctan(k * (s**4 - 2 * s**2)) / np.arctan(k))
    return weight


def compute_contact_forces_wrt_q(
    q: np.ndarray,
    circle_center: np.ndarray,
    circle_radius: float,
    beam_radius: float,
    k_c: float = 1.0,
) -> Tuple[float, np.ndarray, np.ndarray, float, Dict[int, float]]:
    """
    FULL GEOMETRIC MODEL WITH WEIGHTS.

    k_c: contact stiffness multiplier (adapted externally based on force residual).
    """
    N = len(q) // 2
    num_segments = N - 2
    if num_segments <= 0:
        total_force = np.zeros_like(q)
        total_hessian = np.zeros((len(q), len(q)))
        return 0.0, total_force, total_hessian, float("inf"), {}

    nodes = q.reshape(N, 2)

    segment_gaps: Dict[int, float] = {}
    segment_s_values: Dict[int, float] = {}
    weights_values: Dict[int, float] = {}

    total_energy = 0.0
    total_force = np.zeros_like(q)
    total_hessian = np.zeros((len(q), len(q)))

    # IMC parameters and activation distance
    d_imc = 0.0008
    K_imc = 15 / d_imc
    activation_dist = beam_radius + circle_radius + d_imc

    # Get cached JIT-compiled gap gradient/Hessian for this obstacle
    grad_fn_jit, hess_fn_jit = _get_gap_functions_cached(
        circle_center, circle_radius, beam_radius
    )

    for i in range(num_segments):
        segment_id = i + 1
        p1 = nodes[i]
        p2 = nodes[i + 1]
        p3 = nodes[i + 2]

        # Broad-phase culling: if all three nodes are far from the circle,
        # skip expensive geometric/JAX computations.
        min_node_dist = min(
            np.linalg.norm(p1 - circle_center),
            np.linalg.norm(p2 - circle_center),
            np.linalg.norm(p3 - circle_center),
        )
        if min_node_dist > 2.0 * activation_dist:
            continue

        # Full geometric gap with s_i (NumPy only)
        gap, s_i = compute_gap_with_si(p1, p2, p3, circle_center, circle_radius, beam_radius)
        weight = compute_weights_from_s(s_i)

        weights_values[segment_id] = weight
        segment_gaps[segment_id] = gap
        segment_s_values[segment_id] = s_i

        # Use cached JIT-compiled gradient and Hessian
        segment_points = jnp.concatenate([jnp.array(p1), jnp.array(p2), jnp.array(p3)])
        gap_gradient = np.array(grad_fn_jit(segment_points))
        gap_hessian = np.array(hess_fn_jit(segment_points))

        segment_energy = weight * computeE_IMC(gap + beam_radius, beam_radius, d_imc, K_imc)
        total_energy += segment_energy

        force_magnitude = weight * compute_dE_IMC_ddelta_c(
            gap + beam_radius, beam_radius, d_imc, K_imc
        )
        segment_force = force_magnitude * gap_gradient

        outer_product = np.outer(gap_gradient, gap_gradient)
        segment_hessian = weight * (
            compute_d2E_IMC_ddelta_c(gap + beam_radius, beam_radius, d_imc, K_imc)
            * outer_product
            + compute_dE_IMC_ddelta_c(gap + beam_radius, beam_radius, d_imc, K_imc)
            * gap_hessian
        )

        dof_start = 2 * i
        dof_end = 2 * i + 6
        total_force[dof_start:dof_end] += segment_force
        total_hessian[dof_start:dof_end, dof_start:dof_end] += segment_hessian

    # Apply contact stiffness scaling
    total_force *= k_c
    total_hessian *= k_c
    total_energy *= k_c

    # If no segments contributed (all culled as far), report zero forces and infinite gap
    if segment_gaps:
        gap_min = min(segment_gaps.values())
    else:
        gap_min = float("inf")

    return total_energy, total_force, total_hessian, gap_min, weights_values


# =============================================================================
# Dismech wrappers: map 3D DOFs to 2D contact model and back
# =============================================================================


def _update_contact_stiffness(robot, forces_3d, first_iter, ratio=0.9):
    """
    Adaptive contact stiffness: scale k_c so that the contact force is
    comparable to the other forces (elastic + inertia) acting on the beam.

    Uses exponential smoothing so k_c ramps up/down gradually.
    Only updates when called with first_iter=True; the caller
    (compute_contact_forces_wrt_q) will flag whether contact is
    actually active so that k_c is not pre-loaded before contact starts.
    """
    env = robot.env
    if not hasattr(env, '_contact_kc'):
        env._contact_kc = env.imc_stiffness  # initialise from user-supplied kc
        env._contact_active = False           # no contact yet

    if not first_iter:
        return env._contact_kc

    # Compute candidate stiffness from force residual at beam nodes
    node_dof = robot.node_dof_indices  # (n_nodes, 3)
    node_forces = forces_3d[node_dof]  # (n_nodes, 3)
    max_force = np.max(np.linalg.norm(node_forces, axis=1))

    if max_force < 1e-9:
        return env._contact_kc

    new_kc = max_force * env.imc_stiffness
    # Store candidate; actual update happens only if contact is active
    env._contact_kc_candidate = new_kc
    return env._contact_kc


def _finalize_contact_stiffness(robot, has_active_contact, ratio=0.9):
    """
    Called after computing contact forces.  Only ramp up k_c when at least
    one segment is actually in the activation zone, preventing k_c from
    pre-loading before contact starts.
    """
    env = robot.env
    if not hasattr(env, '_contact_kc_candidate'):
        return

    if has_active_contact:
        env._contact_active = True
        env._contact_kc = ratio * env._contact_kc + (1 - ratio) * env._contact_kc_candidate
    elif env._contact_active:
        # Contact was active but no longer — let k_c decay back to base
        env._contact_kc = ratio * env._contact_kc + (1 - ratio) * env.imc_stiffness

    del env._contact_kc_candidate


def compute_circle_four_nodes_imc_weighted_forces_3d(
    robot: "SoftRobot",
    q: np.ndarray,
    circle_center_2d: np.ndarray,
    circle_radius: float,
    beam_radius: float,
    forces_3d: np.ndarray = None,
    first_iter: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[int, float]]:
    """
    Compute weighted IMC circle contact forces for a 2D Dismech rod.

    - Extract (x, z) coordinates from the full 3D node positions in q.
    - Call the 2D contact model (unchanged geometry).
    - Embed the resulting 2D forces/Jacobian back into the 3D DOFs
      (x and z components only; y is untouched, twist DOFs are untouched).
    """
    n_nodes = robot.n_nodes
    node_dof = robot.node_dof_indices  # shape (n_nodes, 3)

    # Adaptive contact stiffness
    k_c = _update_contact_stiffness(robot, forces_3d, first_iter) if forces_3d is not None else getattr(robot.env, '_contact_kc', robot.env.imc_stiffness)

    # Extract node positions and build 2D (x, z) configuration
    node_pos = q[node_dof]  # (n_nodes, 3)
    q2d = np.stack([node_pos[:, 0], node_pos[:, 2]], axis=1).reshape(-1)

    _, f2d, H2d, gap_min, weights = compute_contact_forces_wrt_q(
        q2d, circle_center_2d, circle_radius, beam_radius, k_c=k_c
    )

    # Finalize adaptive stiffness: only ramp up k_c when contact is active
    d_imc = 0.0008
    if first_iter and forces_3d is not None:
        _finalize_contact_stiffness(robot, gap_min < d_imc)

    # Map 2D forces back to 3D DOFs (x,z)
    F = np.zeros_like(q)
    for i in range(n_nodes):
        fx = f2d[2 * i]
        fz = f2d[2 * i + 1]
        F[node_dof[i, 0]] += fx
        F[node_dof[i, 2]] += fz

    # Embed 2D Hessian into 3D DOF space (x,z blocks)
    ndof = q.shape[0]
    J = np.zeros((ndof, ndof))
    for i in range(n_nodes):
        for j in range(n_nodes):
            H_block = H2d[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            if not np.any(H_block):
                continue
            # Map (x,z) of node i/j
            ii_x = node_dof[i, 0]
            ii_z = node_dof[i, 2]
            jj_x = node_dof[j, 0]
            jj_z = node_dof[j, 2]
            J[ii_x, jj_x] += H_block[0, 0]
            J[ii_x, jj_z] += H_block[0, 1]
            J[ii_z, jj_x] += H_block[1, 0]
            J[ii_z, jj_z] += H_block[1, 1]

    return F, J, gap_min, weights


def compute_circle_four_nodes_imc_weighted_forces_3d_xy(
    robot: "SoftRobot",
    q: np.ndarray,
    circle_center_3d: np.ndarray,
    circle_radius: float,
    beam_radius: float,
    forces_3d: np.ndarray = None,
    first_iter: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[int, float]]:
    """
    Variant of the wrapper for a 2D problem in the (x, y) plane:

    - Extract (x, y) coordinates from the full 3D node positions in q.
    - Call the 2D contact model (unchanged geometry).
    - Embed the resulting 2D forces/Jacobian back into the 3D DOFs
      (x and y components only; z is untouched, twist DOFs are untouched).
    """
    n_nodes = robot.n_nodes
    node_dof = robot.node_dof_indices  # shape (n_nodes, 3)

    # Adaptive contact stiffness
    k_c = _update_contact_stiffness(robot, forces_3d, first_iter) if forces_3d is not None else getattr(robot.env, '_contact_kc', robot.env.imc_stiffness)

    # Extract node positions and build 2D (x, y) configuration
    node_pos = q[node_dof]  # (n_nodes, 3)
    q2d = np.stack([node_pos[:, 0], node_pos[:, 1]], axis=1).reshape(-1)

    circle_center_2d = np.asarray(circle_center_3d, dtype=float)[:2]

    _, f2d, H2d, gap_min, weights = compute_contact_forces_wrt_q(
        q2d, circle_center_2d, circle_radius, beam_radius, k_c=k_c
    )

    # Finalize adaptive stiffness: only ramp up k_c when contact is active
    d_imc = 0.0008
    if first_iter and forces_3d is not None:
        _finalize_contact_stiffness(robot, gap_min < d_imc)

    # Map 2D forces back to 3D DOFs (x,y)
    F = np.zeros_like(q)
    for i in range(n_nodes):
        fx = f2d[2 * i]
        fy = f2d[2 * i + 1]
        F[node_dof[i, 0]] += fx
        F[node_dof[i, 1]] += fy

    # Embed 2D Hessian into 3D DOF space (x,y blocks)
    ndof = q.shape[0]
    J = np.zeros((ndof, ndof))
    for i in range(n_nodes):
        for j in range(n_nodes):
            H_block = H2d[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            if not np.any(H_block):
                continue
            # Map (x,y) of node i/j
            ii_x = node_dof[i, 0]
            ii_y = node_dof[i, 1]
            jj_x = node_dof[j, 0]
            jj_y = node_dof[j, 1]
            J[ii_x, jj_x] += H_block[0, 0]
            J[ii_x, jj_y] += H_block[0, 1]
            J[ii_y, jj_x] += H_block[1, 0]
            J[ii_y, jj_y] += H_block[1, 1]

    return F, J, gap_min, weights

