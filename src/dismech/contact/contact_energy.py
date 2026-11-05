import abc
from typing import List

import numpy as np
import sympy as sp

from .contact_pairs import ContactPair
from .. import profiling as _prof


def compute_energy_grad_hess(Delta, delta, h, k_1):
    E = np.zeros_like(Delta)
    grad_E = np.zeros_like(Delta)
    hess_E = np.zeros_like(Delta)

    mask1 = (Delta > 0) & (Delta <= 2 * h - delta)
    mask2 = (Delta > 2 * h - delta) & (Delta < 2 * h + delta)

    # Quadratic region
    E[mask1] = (2 * h - Delta[mask1]) ** 2
    grad_E[mask1] = -2 * (2 * h - Delta[mask1])
    hess_E[mask1] = 2.0

    # Smooth transition region
    exp_term = np.exp(k_1 * (2 * h - Delta[mask2]))
    log_term = np.log(1 + exp_term)
    denom = 1 + exp_term

    E[mask2] = (1 / k_1 * log_term) ** 2
    grad_E[mask2] = -2 * (1 / k_1 * log_term) * (exp_term / denom)
    hess_E[mask2] = (
        2 * exp_term / (denom ** 2) *
        (log_term + exp_term)
    ) # corrected

    return E, grad_E, hess_E

class ContactEnergy(metaclass=abc.ABCMeta):

    def __init__(self, pairs: List[ContactPair], delta: float, h: float, k_1: float, stiffness: float, scale: bool = True):
        self._pairs = np.vstack([p.pair_nodes for p in pairs])
        self.original_k_scalar = stiffness
        self.k_c = stiffness
        self._ind = np.vstack([p.ind for p in pairs])   # total inds, mask during first iter
        if scale:
            self.scale = 1.0 / h
        else:
            self.scale = 1.0

        Delta = sp.symbols("Delta")
        norm_delta = delta * self.scale
        norm_k_1 = k_1 / self.scale
        norm_h = h * self.scale

        self.norm_delta = delta * self.scale
        self.norm_h = h * self.scale
        self.norm_k_1 = k_1 / self.scale

        # debug:
        print("delta:", self.norm_delta)
        print("h:", self.norm_h)
        print("K1:", self.norm_k_1)
        print("scale: ", self.scale)
        print("kc: ", self.k_c)

        print("upper limit for quadratic:", 2 * self.norm_h - self.norm_delta)
        print("upper limit for smooth:", 2 * self.norm_h + self.norm_delta)
        
        # self.__expr = get_E(Delta, norm_delta, norm_h, norm_k_1)
        # grad_expr = get_grad_E(Delta, norm_delta, norm_h, norm_k_1)
        # hess_expr = get_hess_E(Delta, norm_delta, norm_h, norm_k_1)

        # self.__fn = sp.lambdify(Delta, self.__expr, modules='numpy')
        # self.__grad_fn = sp.lambdify(Delta, grad_expr, modules='numpy')
        # self.__hess_fn = sp.lambdify(Delta, hess_expr, modules='numpy')

        # self.__fn = sp.lambdify(Delta, self.__expr, modules='numpy')
        # self.__grad_fn = sp.lambdify(Delta, sp.diff(
        #     self.__expr, Delta), modules='numpy')
        # self.__hess_fn = sp.lambdify(Delta, sp.diff(
        #     self.__expr, Delta, Delta), modules='numpy')

    def get_energy(self, q, output_scalar: bool = True):
        # Always check all indices
        self.ind = self._ind
        self.pairs = self._pairs
        q = q * self.scale
        Delta = self.get_Delta(q)
        E, _, _ = compute_energy_grad_hess(Delta, self.norm_delta, self.norm_h, self.norm_k_1)
        return np.sum(E) if output_scalar else E

    def grad_hess_energy(self, state, robot, F, first_iter):
        q = state.q
        q = q * self.scale

        # Broad phase: the candidate filter that fixes the active pair list
        # for the whole time step (recomputed only on the first iteration).
        if first_iter:
            with _prof.phase('broad'):
                self.ind = self._ind
                self.pairs = self._pairs
                first_Delta = self.get_Delta(q)
                self.ind = self._ind[first_Delta < 2 * self.norm_h + self.norm_delta * 10]
                self.pairs = self._pairs[first_Delta < 2 * self.norm_h + self.norm_delta * 10]
            _prof.count('pairs_stencil', self._ind.shape[0])

            self.get_contact_stiffness(robot, F)

        _prof.count('contact_calls')
        _prof.count('pairs_candidate', self.ind.shape[0])

        if self.ind.size == 0:
            n_dof = q.shape[0]
            return np.zeros(n_dof), np.zeros((n_dof, n_dof))

        # Narrow phase: distance query and its derivatives on the candidates.
        with _prof.phase('narrow'):
            Delta = self.get_Delta(q)
            grad_Delta, hess_Delta = self.get_grad_hess_Delta(q)

            E, grad_E_D, hess_E_D = compute_energy_grad_hess(Delta, self.norm_delta, self.norm_h, self.norm_k_1)

            grad_E = grad_Delta * grad_E_D[:, None]
            hess_E = hess_E_D[:, None, None] * np.einsum('ni,nj->nij', grad_Delta, grad_Delta)
            hess_E += grad_E_D[:, None, None] * hess_Delta

            grad_E *= self.scale * self.k_c
            hess_E *= self.scale ** 2 * self.k_c
        _prof.count('pairs_active', int(np.count_nonzero(grad_E_D)))

        n_dof = q.shape[0]

        # Assembly: scatter into the global force vector and Jacobian.
        with _prof.phase('assembly'):
            Fs = np.zeros(n_dof)
            np.add.at(Fs, self.ind, -grad_E)
            Js = np.zeros((n_dof, n_dof))
            np.add.at(Js, (self.ind[:, :, None], self.ind[:, None, :]), -hess_E)

        # Cache per-pair contact-force gradient for downstream consumers
        # (notably IMCFrictionEnergy, which needs the PER-PAIR nodal force
        # to weight friction correctly — reading the GLOBAL contact_force
        # vector and indexing by self.ind double-counts nodes that appear
        # in multiple pairs).  Shape: (n_pairs, 12).  Sign matches the
        # physical contact force on the body (== -grad_E in the same way
        # the assembled Fs uses -grad_E).
        self._last_per_pair_force = -grad_E.copy()
        # Pair indices the cached force corresponds to (same self.ind row
        # ordering — stored so the consumer can sanity-check it didn't
        # change between calls).
        self._last_per_pair_ind = self.ind

        # return Fs, Js
        return Fs, Js
    
    def get_contact_stiffness(self, robot, F: np.ndarray, ):
        ratio = 0.9
        if np.sum(np.abs(F)) < 1e-9:
            return
        # Scale contact stiffness by the max nodal force in the system
        # (not just at contact pair nodes) so that contact forces can
        # match bending/stretch forces everywhere.
        node_forces = F[:robot.end_node_dof_index].reshape(-1, 3)
        max_force = np.max(np.linalg.norm(node_forces, axis=1))
        self.k_c = ratio*self.k_c + (1 - ratio) * max_force * self.original_k_scalar
        return

    @abc.abstractmethod
    def get_Delta(self, q):
        pass

    @abc.abstractmethod
    def get_grad_hess_Delta(self, q):
        pass
