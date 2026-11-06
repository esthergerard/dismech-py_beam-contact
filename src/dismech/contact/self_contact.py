"""
Self-contact (beam-to-beam) force computation using arc-to-arc gap geometry.

Uses the same IMC energy and weight functions as the circle contact module.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import logsumexp

from .circle_four_nodes_imc_weighted import (
    computeE_IMC,
    compute_dE_IMC_ddelta_c,
    compute_d2E_IMC_ddelta_c,
    compute_curvature_jax,
    fix_arc_angles_jax,
    compute_weights_from_s,
)


# =============================================================================
# Arc-to-arc gap computation (JAX-differentiable)
# =============================================================================

def compute_arc_to_arc_gap_jax(p1, p2, p3, p11, p22, p33, beam_radius1, beam_radius2):
    """
    Computes the gap and s_i values using a physically-motivated, penalized blending model.

    Fits circular arcs to both 3-node segments, computes 4 candidate closest-point
    pairs (endpoint combos of extended arcs), uses smin (smooth min with LogSumExp)
    over the 4 candidate distances, and softmax to blend s_i values.

    Returns (gap, s_i1, s_i2).
    """
    def smin(a, alpha=0.01):
        return -alpha * logsumexp(-jnp.array(a) / alpha)

    def softmax(x, temp=0.05):
        x_stable = x - jnp.max(x)
        exp_x = jnp.exp(x_stable / temp)
        return exp_x / jnp.sum(exp_x)

    # --- Arc 1: fit circle through p1, p2, p3 ---
    kappa1 = compute_curvature_jax(p1, p2, p3)
    kappa1 = jnp.where(jnp.abs(kappa1) < 1e-3, jnp.where(kappa1 >= 0, 1e-3, -1e-3), kappa1)
    R_beam1 = 1.0 / kappa1
    R_curv1 = jnp.abs(R_beam1)
    mid1_1, mid2_1 = (p1 + p2) / 2, (p2 + p3) / 2
    chord_midpoint1 = (mid1_1 + mid2_1) / 2
    chord_vector1 = mid2_1 - mid1_1
    chord_length1 = jnp.linalg.norm(chord_vector1)
    perp_direction1 = jnp.array([-chord_vector1[1], chord_vector1[0]]) / jnp.maximum(chord_length1, 1e-12)
    d_squared1 = jnp.maximum(R_beam1**2 - (chord_length1 / 2)**2, 1e-12)
    d1 = jnp.sqrt(d_squared1)
    C_beam1 = jnp.where(kappa1 > 0, chord_midpoint1 + d1 * perp_direction1, chord_midpoint1 - d1 * perp_direction1)
    angle_mid1_1 = jnp.arctan2(mid1_1[1] - C_beam1[1], mid1_1[0] - C_beam1[0])
    angle_mid2_1 = jnp.arctan2(mid2_1[1] - C_beam1[1], mid2_1[0] - C_beam1[0])
    angle_mid1_1, angle_mid2_1 = fix_arc_angles_jax(angle_mid1_1, angle_mid2_1, kappa1, C_beam1, mid1_1, mid2_1, R_curv1)
    original_angle_span1 = jnp.where(kappa1 > 0, angle_mid2_1 - angle_mid1_1, angle_mid1_1 - angle_mid2_1)
    original_angle_span1 = jnp.where(original_angle_span1 < 0, original_angle_span1 + 2 * jnp.pi, original_angle_span1)
    arc_extension1 = jnp.maximum(original_angle_span1, 1e-6) / 2
    angle_mid1_ext1 = jnp.where(kappa1 > 0, angle_mid1_1 - arc_extension1, angle_mid1_1 + arc_extension1)
    angle_mid2_ext1 = jnp.where(kappa1 > 0, angle_mid2_1 + arc_extension1, angle_mid2_1 - arc_extension1)

    # --- Arc 2: fit circle through p11, p22, p33 ---
    kappa2 = compute_curvature_jax(p11, p22, p33)
    kappa2 = jnp.where(jnp.abs(kappa2) < 1e-3, jnp.where(kappa2 >= 0, 1e-3, -1e-3), kappa2)
    R_beam2 = 1.0 / kappa2
    R_curv2 = jnp.abs(R_beam2)
    mid1_2, mid2_2 = (p11 + p22) / 2, (p22 + p33) / 2
    chord_midpoint2 = (mid1_2 + mid2_2) / 2
    chord_vector2 = mid2_2 - mid1_2
    chord_length2 = jnp.linalg.norm(chord_vector2)
    perp_direction2 = jnp.array([-chord_vector2[1], chord_vector2[0]]) / jnp.maximum(chord_length2, 1e-12)
    d_squared2 = jnp.maximum(R_beam2**2 - (chord_length2 / 2)**2, 1e-12)
    d2 = jnp.sqrt(d_squared2)
    C_beam2 = jnp.where(kappa2 > 0, chord_midpoint2 + d2 * perp_direction2, chord_midpoint2 - d2 * perp_direction2)
    angle_mid1_2 = jnp.arctan2(mid1_2[1] - C_beam2[1], mid1_2[0] - C_beam2[0])
    angle_mid2_2 = jnp.arctan2(mid2_2[1] - C_beam2[1], mid2_2[0] - C_beam2[0])
    angle_mid1_2, angle_mid2_2 = fix_arc_angles_jax(angle_mid1_2, angle_mid2_2, kappa2, C_beam2, mid1_2, mid2_2, R_curv2)
    original_angle_span2 = jnp.where(kappa2 > 0, angle_mid2_2 - angle_mid1_2, angle_mid1_2 - angle_mid2_2)
    original_angle_span2 = jnp.where(original_angle_span2 < 0, original_angle_span2 + 2 * jnp.pi, original_angle_span2)
    arc_extension2 = jnp.maximum(original_angle_span2, 1e-6) / 2
    angle_mid1_ext2 = jnp.where(kappa2 > 0, angle_mid1_2 - arc_extension2, angle_mid1_2 + arc_extension2)
    angle_mid2_ext2 = jnp.where(kappa2 > 0, angle_mid2_2 + arc_extension2, angle_mid2_2 - arc_extension2)

    # --- Compute 4 candidate endpoint pairs ---
    end1_start = C_beam1 + R_curv1 * jnp.array([jnp.cos(angle_mid1_ext1), jnp.sin(angle_mid1_ext1)])
    end1_end   = C_beam1 + R_curv1 * jnp.array([jnp.cos(angle_mid2_ext1), jnp.sin(angle_mid2_ext1)])
    end2_start = C_beam2 + R_curv2 * jnp.array([jnp.cos(angle_mid1_ext2), jnp.sin(angle_mid1_ext2)])
    end2_end   = C_beam2 + R_curv2 * jnp.array([jnp.cos(angle_mid2_ext2), jnp.sin(angle_mid2_ext2)])

    candidate_distances = jnp.array([
        jnp.linalg.norm(end1_start - end2_start),
        jnp.linalg.norm(end1_start - end2_end),
        jnp.linalg.norm(end1_end - end2_start),
        jnp.linalg.norm(end1_end - end2_end)
    ])
    all_candidate_gaps = candidate_distances - beam_radius1 - beam_radius2

    # Since all candidates are endpoints, their s-values are fixed.
    candidate_s1_values = jnp.array([-1.0, -1.0, 1.0, 1.0])
    candidate_s2_values = jnp.array([-1.0, 1.0, -1.0, 1.0])

    # All candidates are valid by definition, so no penalties are needed.
    penalized_gaps = all_candidate_gaps

    # Compute the final gap directly using smin.
    gap = smin(penalized_gaps)

    # Compute the blending weights using softmax.
    weights = softmax(-penalized_gaps)

    # Compute the blended s_i values using a simple weighted average.
    s_i1 = jnp.sum(weights * candidate_s1_values)
    s_i2 = jnp.sum(weights * candidate_s2_values)

    return gap, s_i1, s_i2


def compute_edge_to_edge_gap_jax(p1, p2, p3, p11, p22, p33, beam_radius1, beam_radius2):
    """
    Edge-to-edge closest-point gap for 3-node segments (2D).

    Each segment is represented by the line between its two edge midpoints:
      segment 1: mid1 = (p1+p2)/2  to  mid2 = (p2+p3)/2
      segment 2: mid3 = (p11+p22)/2  to  mid4 = (p22+p33)/2

    Uses a smooth Lumelsky-like parametrization with clamping via LogSumExp
    to avoid derivative discontinuities at segment endpoints.
    """
    # Scale smoothing parameters by beam radius so they remain small
    # relative to the physical gap values.
    r_ref = jnp.maximum(beam_radius1, beam_radius2)
    smin_alpha = 0.1 * r_ref
    softmax_temp = 0.1 * r_ref

    def smin(a):
        return -smin_alpha * logsumexp(-jnp.array(a) / smin_alpha)

    def softmax(x):
        x_stable = x - jnp.max(x)
        exp_x = jnp.exp(x_stable / softmax_temp)
        return exp_x / jnp.sum(exp_x)

    # Edge midpoints
    a1 = (p1 + p2) / 2
    a2 = (p2 + p3) / 2
    b1 = (p11 + p22) / 2
    b2 = (p22 + p33) / 2

    # 4 endpoint-pair distances (same structure as arc model)
    candidate_distances = jnp.array([
        jnp.linalg.norm(a1 - b1),
        jnp.linalg.norm(a1 - b2),
        jnp.linalg.norm(a2 - b1),
        jnp.linalg.norm(a2 - b2),
    ])
    all_candidate_gaps = candidate_distances - beam_radius1 - beam_radius2

    candidate_s1_values = jnp.array([-1.0, -1.0, 1.0, 1.0])
    candidate_s2_values = jnp.array([-1.0, 1.0, -1.0, 1.0])

    gap = smin(all_candidate_gaps)
    weights = softmax(-all_candidate_gaps)
    s_i1 = jnp.sum(weights * candidate_s1_values)
    s_i2 = jnp.sum(weights * candidate_s2_values)

    return gap, s_i1, s_i2


def compute_hybrid_gap_jax(p1, p2, p3, p11, p22, p33, beam_radius1, beam_radius2, kappa_threshold=0.5):
    """
    Hybrid gap: smoothly blends between arc-to-arc and edge-to-edge models
    based on the curvature of each segment.

    When both segments have |kappa| > kappa_threshold, uses arc-to-arc.
    When either segment has |kappa| < kappa_threshold, blends toward edge-to-edge.
    The blending uses a smooth sigmoid to avoid derivative jumps.
    """
    kappa1 = compute_curvature_jax(p1, p2, p3)
    kappa2 = compute_curvature_jax(p11, p22, p33)

    # Smooth sigmoid blend: alpha = 1 means arc, alpha = 0 means edge
    # Use the minimum |kappa| of the two segments
    min_abs_kappa = jnp.minimum(jnp.abs(kappa1), jnp.abs(kappa2))
    steepness = 20.0 / kappa_threshold  # transition over ~10% of threshold
    alpha = jax.nn.sigmoid(steepness * (min_abs_kappa - kappa_threshold))

    gap_arc, s1_arc, s2_arc = compute_arc_to_arc_gap_jax(
        p1, p2, p3, p11, p22, p33, beam_radius1, beam_radius2)
    gap_edge, s1_edge, s2_edge = compute_edge_to_edge_gap_jax(
        p1, p2, p3, p11, p22, p33, beam_radius1, beam_radius2)

    gap = alpha * gap_arc + (1 - alpha) * gap_edge
    s_i1 = alpha * s1_arc + (1 - alpha) * s1_edge
    s_i2 = alpha * s2_arc + (1 - alpha) * s2_edge

    return gap, s_i1, s_i2


def compute_extended_arc_to_arc_gap_jax_smooth(p1, p2, p3, p11, p22, p33, beam_radius1, beam_radius2):
    """Wrapper that calls compute_arc_to_arc_gap_jax."""
    return compute_arc_to_arc_gap_jax(p1, p2, p3, p11, p22, p33, beam_radius1, beam_radius2)


# =============================================================================
# Hessian reconstruction
# =============================================================================

def reconstruct_hessian_from_jax(hess_parts):
    """
    Converts the nested tuple output of jax.hessian into a dense numpy matrix.
    """
    num_points = 6  # p1, p2, p3, p11, p22, p33
    H = np.zeros((12, 12))
    for row_idx in range(num_points):
        for col_idx in range(num_points):
            block = hess_parts[row_idx][col_idx]
            H[row_idx*2:(row_idx+1)*2, col_idx*2:(col_idx+1)*2] = block
    return H


# =============================================================================
# Analytic derivatives of weights
# =============================================================================

def compute_dw_ds(s):
    k = 0.8204
    ds_term = 4 * s**3 - 4 * s
    denominator = 1 + (k * (s**4 - 2 * s**2))**2
    return (k * ds_term) / (jnp.arctan(k) * denominator)


def compute_d2w_ds2(s):
    k = 0.8204
    s2, s4, atan_k = s**2, s**4, jnp.arctan(k)
    u = k * (s4 - 2 * s2)
    du_ds = k * (4 * s**3 - 4 * s)
    d2u_ds2 = k * (12 * s2 - 4)
    term1 = d2u_ds2 * (1 + u**2)
    term2 = du_ds * (2 * u * du_ds)
    denominator = (1 + u**2)**2
    return (1 / atan_k) * (term1 - term2) / denominator


# =============================================================================
# JIT-compiled derivative function
# =============================================================================

@partial(jax.jit, static_argnums=(6, 7))
def _compute_all_derivatives_jit(p1, p2, p3, p11, p22, p33, r1, r2):
    """
    A JIT-compiled function to compute all geometric values and their derivatives
    using the fully smooth gap function.
    """
    def get_gap(*points):
        return compute_extended_arc_to_arc_gap_jax_smooth(*points, r1, r2)[0]

    def get_s1(*points):
        return compute_extended_arc_to_arc_gap_jax_smooth(*points, r1, r2)[1]

    def get_s2(*points):
        return compute_extended_arc_to_arc_gap_jax_smooth(*points, r1, r2)[2]

    points_jax = (p1, p2, p3, p11, p22, p33)

    gap, grad_gap_parts = jax.value_and_grad(get_gap, argnums=range(6))(*points_jax)
    hess_gap_parts = jax.hessian(get_gap, argnums=range(6))(*points_jax)

    s_i1, grad_s1_parts = jax.value_and_grad(get_s1, argnums=range(6))(*points_jax)
    hess_s1_parts = jax.hessian(get_s1, argnums=range(6))(*points_jax)

    s_i2, grad_s2_parts = jax.value_and_grad(get_s2, argnums=range(6))(*points_jax)
    hess_s2_parts = jax.hessian(get_s2, argnums=range(6))(*points_jax)

    return (gap, s_i1, s_i2,
            grad_gap_parts, hess_gap_parts,
            grad_s1_parts, hess_s1_parts,
            grad_s2_parts, hess_s2_parts)


@partial(jax.jit, static_argnums=(6, 7, 8))
def _compute_all_derivatives_hybrid_jit(p1, p2, p3, p11, p22, p33, r1, r2, kappa_threshold_x1000):
    """
    JIT-compiled derivative function using the hybrid arc/edge model.
    kappa_threshold is passed as int (x1000) to be a valid static arg.
    """
    kt = kappa_threshold_x1000 / 1000.0

    def get_gap(*points):
        return compute_hybrid_gap_jax(*points, r1, r2, kt)[0]

    def get_s1(*points):
        return compute_hybrid_gap_jax(*points, r1, r2, kt)[1]

    def get_s2(*points):
        return compute_hybrid_gap_jax(*points, r1, r2, kt)[2]

    points_jax = (p1, p2, p3, p11, p22, p33)

    gap, grad_gap_parts = jax.value_and_grad(get_gap, argnums=range(6))(*points_jax)
    hess_gap_parts = jax.hessian(get_gap, argnums=range(6))(*points_jax)

    s_i1, grad_s1_parts = jax.value_and_grad(get_s1, argnums=range(6))(*points_jax)
    hess_s1_parts = jax.hessian(get_s1, argnums=range(6))(*points_jax)

    s_i2, grad_s2_parts = jax.value_and_grad(get_s2, argnums=range(6))(*points_jax)
    hess_s2_parts = jax.hessian(get_s2, argnums=range(6))(*points_jax)

    return (gap, s_i1, s_i2,
            grad_gap_parts, hess_gap_parts,
            grad_s1_parts, hess_s1_parts,
            grad_s2_parts, hess_s2_parts)


# =============================================================================
# Self-contact force computation
# =============================================================================

def compute_self_contact_forces(q, beam_radius, k_c=1.0, min_segment_separation=6):
    """
    Computes self-contact forces within a single beam.

    Loops over segment pairs (i, j) with |i - j| >= min_segment_separation,
    uses broad-phase culling with bounding spheres, computes arc-to-arc gap
    and IMC energy with weighted partition of unity.

    Parameters
    ----------
    q : (2*N,) array -- 2D node coordinates [x0, y0, x1, y1, ...].
    beam_radius : float -- cross-section radius.
    k_c : float -- contact stiffness multiplier.
    min_segment_separation : int -- minimum index gap between segment pairs.

    Returns
    -------
    total_energy, total_force, total_hessian, all_gaps, {}
    """
    N = len(q) // 2
    num_segments = N - 2
    total_dof = N * 2

    if num_segments <= 1:
        return 0.0, np.zeros(total_dof), np.zeros((total_dof, total_dof)), [], {}

    nodes = q.reshape(N, 2)
    total_energy = 0.0
    total_force = np.zeros(total_dof)
    total_hessian = np.zeros((total_dof, total_dof))
    all_gaps = []

    beam_radius_scalar = float(beam_radius)
    d_imc_scalar = 0.008
    K_imc_scalar = 15.0 / d_imc_scalar

    for i in range(num_segments):
        for j in range(i + min_segment_separation, num_segments):
            p1, p2, p3 = nodes[i:i+3]
            p11, p22, p33 = nodes[j:j+3]

            # Broad-phase: bounding sphere culling
            segment_i_center = (nodes[i] + nodes[i+2]) / 2
            segment_j_center = (nodes[j] + nodes[j+2]) / 2
            center_distance = np.linalg.norm(segment_j_center - segment_i_center)

            r_seg_i = np.linalg.norm(nodes[i] - nodes[i+2]) / 2
            r_seg_j = np.linalg.norm(nodes[j] - nodes[j+2]) / 2

            interaction_range = 2 * beam_radius_scalar + d_imc_scalar
            safe_distance = interaction_range + r_seg_i + r_seg_j

            if center_distance > safe_distance:
                continue

            # Compute contact using arc-to-arc computation
            (gap, s_i1, s_i2,
             grad_gap_parts, hess_gap_parts,
             grad_s1_parts, hess_s1_parts,
             grad_s2_parts, hess_s2_parts) = _compute_all_derivatives_jit(
                jnp.array(p1), jnp.array(p2), jnp.array(p3),
                jnp.array(p11), jnp.array(p22), jnp.array(p33),
                beam_radius_scalar, beam_radius_scalar)

            gap_info = {
                'segment_pair': f"({i},{j})",
                'gap': float(gap),
                's_i1': float(s_i1),
                's_i2': float(s_i2),
                'beam_type': 'self_contact'
            }
            all_gaps.append(gap_info)

            # Convert JAX outputs to flat numpy arrays/matrices
            grad_gap_vec = np.concatenate([np.asarray(g).flatten() for g in grad_gap_parts])
            grad_s1_vec = np.concatenate([np.asarray(g).flatten() for g in grad_s1_parts])
            grad_s2_vec = np.concatenate([np.asarray(g).flatten() for g in grad_s2_parts])

            H_gap = reconstruct_hessian_from_jax(hess_gap_parts)
            H_s1 = reconstruct_hessian_from_jax(hess_s1_parts)
            H_s2 = reconstruct_hessian_from_jax(hess_s2_parts)

            H_gap = np.nan_to_num(H_gap)
            H_s1 = np.nan_to_num(H_s1)
            H_s2 = np.nan_to_num(H_s2)

            # Analytic derivatives of weights and energy
            w1 = compute_weights_from_s(float(s_i1))
            w2 = compute_weights_from_s(float(s_i2))
            dw1_ds1 = float(compute_dw_ds(s_i1))
            dw2_ds2 = float(compute_dw_ds(s_i2))
            d2w1_ds1_2 = float(compute_d2w_ds2(s_i1))
            d2w2_ds2_2 = float(compute_d2w_ds2(s_i2))

            # Boundary corrections: beam endpoints get full weight
            if i == 0 and s_i1 < 0:
                w1, dw1_ds1, d2w1_ds1_2 = 1.0, 0.0, 0.0
            if i == num_segments - 1 and s_i1 > 0:
                w1, dw1_ds1, d2w1_ds1_2 = 1.0, 0.0, 0.0
            if j == 0 and s_i2 < 0:
                w2, dw2_ds2, d2w2_ds2_2 = 1.0, 0.0, 0.0
            if j == num_segments - 1 and s_i2 > 0:
                w2, dw2_ds2, d2w2_ds2_2 = 1.0, 0.0, 0.0

            # IMC energy
            delta_c = float(gap) + beam_radius_scalar
            U = computeE_IMC(delta_c, beam_radius_scalar, d_imc_scalar, K_imc_scalar)
            dU_dgap = compute_dE_IMC_ddelta_c(delta_c, beam_radius_scalar, d_imc_scalar, K_imc_scalar)
            d2U_dgap2 = compute_d2E_IMC_ddelta_c(delta_c, beam_radius_scalar, d_imc_scalar, K_imc_scalar)

            # E = w1 * w2 * U
            segment_energy = w1 * w2 * U
            total_energy += segment_energy

            # Force: grad E = grad(w1)*w2*U + w1*grad(w2)*U + w1*w2*grad(U)
            grad_w1 = dw1_ds1 * grad_s1_vec
            grad_w2 = dw2_ds2 * grad_s2_vec
            grad_U = dU_dgap * grad_gap_vec

            segment_force = (grad_w1 * w2 * U) + (w1 * grad_w2 * U) + (w1 * w2 * grad_U)

            # Hessian: positive-definite terms only for stability
            H_term1 = w1 * w2 * d2U_dgap2 * np.outer(grad_gap_vec, grad_gap_vec)
            H_term2 = w2 * U * d2w1_ds1_2 * np.outer(grad_s1_vec, grad_s1_vec)
            H_term3 = w1 * U * d2w2_ds2_2 * np.outer(grad_s2_vec, grad_s2_vec)

            segment_hessian = H_term1 + H_term2 + H_term3

            # Map local 12 DOFs to global DOFs
            global_indices = np.array([
                2*i, 2*i+1,
                2*(i+1), 2*(i+1)+1,
                2*(i+2), 2*(i+2)+1,
                2*j, 2*j+1,
                2*(j+1), 2*(j+1)+1,
                2*(j+2), 2*(j+2)+1
            ])

            total_force[global_indices] += segment_force
            total_hessian[np.ix_(global_indices, global_indices)] += segment_hessian

    total_force *= k_c
    total_hessian *= k_c
    total_energy *= k_c

    return total_energy, total_force, total_hessian, all_gaps, {}


def compute_beam_to_beam_contact_forces(q, n_nodes_A, beam_radius, k_c=1.0, d_imc=None,
                                         kappa_threshold=None):
    """
    Computes contact forces between two separate beams A and B.

    The combined 2D DOF vector q contains beam A nodes first, then beam B:
        q = [xA0, yA0, xA1, yA1, ..., xB0, yB0, xB1, yB1, ...]

    Loops over all segment pairs (i from A, j from B), using the same
    arc-to-arc gap and weighted IMC energy as compute_self_contact_forces.

    Parameters
    ----------
    q : (2*(N_A + N_B),) array -- 2D node coordinates.
    n_nodes_A : int -- number of nodes in beam A.
    beam_radius : float -- cross-section radius (same for both beams).
    k_c : float -- contact stiffness multiplier.

    Returns
    -------
    total_energy, total_force, total_hessian, all_gaps, {}
    """
    N_total = len(q) // 2
    N_B = N_total - n_nodes_A
    num_seg_A = n_nodes_A - 2
    num_seg_B = N_B - 2
    total_dof = N_total * 2

    if num_seg_A <= 0 or num_seg_B <= 0:
        return 0.0, np.zeros(total_dof), np.zeros((total_dof, total_dof)), [], {}

    nodes = q.reshape(N_total, 2)
    nodes_A = nodes[:n_nodes_A]
    nodes_B = nodes[n_nodes_A:]

    total_energy = 0.0
    total_force = np.zeros(total_dof)
    total_hessian = np.zeros((total_dof, total_dof))
    all_gaps = []

    beam_radius_scalar = float(beam_radius)
    if d_imc is None:
        d_imc_scalar = 0.008
    else:
        d_imc_scalar = float(d_imc)
    K_imc_scalar = 15.0 / d_imc_scalar

    for i in range(num_seg_A):
        for j in range(num_seg_B):
            p1, p2, p3 = nodes_A[i:i+3]
            p11, p22, p33 = nodes_B[j:j+3]

            # Broad-phase: bounding sphere culling
            seg_i_center = (nodes_A[i] + nodes_A[i+2]) / 2
            seg_j_center = (nodes_B[j] + nodes_B[j+2]) / 2
            center_distance = np.linalg.norm(seg_j_center - seg_i_center)

            r_seg_i = np.linalg.norm(nodes_A[i] - nodes_A[i+2]) / 2
            r_seg_j = np.linalg.norm(nodes_B[j] - nodes_B[j+2]) / 2

            interaction_range = 2 * beam_radius_scalar + d_imc_scalar
            safe_distance = interaction_range + r_seg_i + r_seg_j

            if center_distance > safe_distance:
                continue

            # Compute contact gap and derivatives
            if kappa_threshold is not None:
                # Hybrid model: blend arc-to-arc with edge-to-edge for low curvature
                kt_int = int(round(kappa_threshold * 1000))
                (gap, s_i1, s_i2,
                 grad_gap_parts, hess_gap_parts,
                 grad_s1_parts, hess_s1_parts,
                 grad_s2_parts, hess_s2_parts) = _compute_all_derivatives_hybrid_jit(
                    jnp.array(p1), jnp.array(p2), jnp.array(p3),
                    jnp.array(p11), jnp.array(p22), jnp.array(p33),
                    beam_radius_scalar, beam_radius_scalar, kt_int)
            else:
                # Pure arc-to-arc model
                (gap, s_i1, s_i2,
                 grad_gap_parts, hess_gap_parts,
                 grad_s1_parts, hess_s1_parts,
                 grad_s2_parts, hess_s2_parts) = _compute_all_derivatives_jit(
                    jnp.array(p1), jnp.array(p2), jnp.array(p3),
                    jnp.array(p11), jnp.array(p22), jnp.array(p33),
                    beam_radius_scalar, beam_radius_scalar)

            gap_info = {
                'segment_pair': f"(A{i},B{j})",
                'gap': float(gap),
                's_i1': float(s_i1),
                's_i2': float(s_i2),
                'beam_type': 'beam_to_beam'
            }
            all_gaps.append(gap_info)

            # Convert JAX outputs to flat numpy arrays/matrices
            grad_gap_vec = np.concatenate([np.asarray(g).flatten() for g in grad_gap_parts])
            grad_s1_vec = np.concatenate([np.asarray(g).flatten() for g in grad_s1_parts])
            grad_s2_vec = np.concatenate([np.asarray(g).flatten() for g in grad_s2_parts])

            H_gap = reconstruct_hessian_from_jax(hess_gap_parts)
            H_s1 = reconstruct_hessian_from_jax(hess_s1_parts)
            H_s2 = reconstruct_hessian_from_jax(hess_s2_parts)

            H_gap = np.nan_to_num(H_gap)
            H_s1 = np.nan_to_num(H_s1)
            H_s2 = np.nan_to_num(H_s2)

            # Analytic derivatives of weights and energy
            w1 = compute_weights_from_s(float(s_i1))
            w2 = compute_weights_from_s(float(s_i2))
            dw1_ds1 = float(compute_dw_ds(s_i1))
            dw2_ds2 = float(compute_dw_ds(s_i2))
            d2w1_ds1_2 = float(compute_d2w_ds2(s_i1))
            d2w2_ds2_2 = float(compute_d2w_ds2(s_i2))

            # Boundary corrections: beam endpoints get full weight
            if i == 0 and s_i1 < 0:
                w1, dw1_ds1, d2w1_ds1_2 = 1.0, 0.0, 0.0
            if i == num_seg_A - 1 and s_i1 > 0:
                w1, dw1_ds1, d2w1_ds1_2 = 1.0, 0.0, 0.0
            if j == 0 and s_i2 < 0:
                w2, dw2_ds2, d2w2_ds2_2 = 1.0, 0.0, 0.0
            if j == num_seg_B - 1 and s_i2 > 0:
                w2, dw2_ds2, d2w2_ds2_2 = 1.0, 0.0, 0.0

            # IMC energy — scale by 1/beam_radius (same as non-regularized
            # scale=1/h) to get comparable force magnitudes.
            scale = 1.0 / beam_radius_scalar
            delta_c = (float(gap) + beam_radius_scalar) * scale
            R_scaled = beam_radius_scalar * scale  # = 1.0
            d_imc_scaled = d_imc_scalar * scale
            K_imc_local = 15.0 / d_imc_scaled
            U = computeE_IMC(delta_c, R_scaled, d_imc_scaled, K_imc_local)
            dU_dgap = compute_dE_IMC_ddelta_c(delta_c, R_scaled, d_imc_scaled, K_imc_local) * scale
            d2U_dgap2 = compute_d2E_IMC_ddelta_c(delta_c, R_scaled, d_imc_scaled, K_imc_local) * scale**2

            # E = w1 * w2 * U
            segment_energy = w1 * w2 * U
            total_energy += segment_energy

            # Force: grad E = grad(w1)*w2*U + w1*grad(w2)*U + w1*w2*grad(U)
            grad_w1 = dw1_ds1 * grad_s1_vec
            grad_w2 = dw2_ds2 * grad_s2_vec
            grad_U = dU_dgap * grad_gap_vec

            segment_force = (grad_w1 * w2 * U) + (w1 * grad_w2 * U) + (w1 * w2 * grad_U)

            # Hessian: positive-definite terms only for stability
            H_term1 = w1 * w2 * d2U_dgap2 * np.outer(grad_gap_vec, grad_gap_vec)
            H_term2 = w2 * U * d2w1_ds1_2 * np.outer(grad_s1_vec, grad_s1_vec)
            H_term3 = w1 * U * d2w2_ds2_2 * np.outer(grad_s2_vec, grad_s2_vec)

            segment_hessian = H_term1 + H_term2 + H_term3

            # Map local 12 DOFs to global DOFs
            # Beam A segment i: nodes i, i+1, i+2 (global 2D DOFs)
            # Beam B segment j: nodes n_nodes_A+j, n_nodes_A+j+1, n_nodes_A+j+2
            gj = n_nodes_A + j
            global_indices = np.array([
                2*i, 2*i+1,
                2*(i+1), 2*(i+1)+1,
                2*(i+2), 2*(i+2)+1,
                2*gj, 2*gj+1,
                2*(gj+1), 2*(gj+1)+1,
                2*(gj+2), 2*(gj+2)+1,
            ])

            total_force[global_indices] += segment_force
            total_hessian[np.ix_(global_indices, global_indices)] += segment_hessian

    total_force *= k_c
    total_hessian *= k_c
    total_energy *= k_c

    return total_energy, total_force, total_hessian, all_gaps, {}
