import numpy as np


class Environment:

    def __init__(self):
        self.__ext_force_list = []

    def add_force(self, key, **kwargs) -> None:
        if key == 'gravity':
            self.g = kwargs['g']
        elif key == 'buoyancy':
            self.rho = kwargs['rho']
        elif key == 'rft':
            self.ct = kwargs['ct']
            self.cn = kwargs['cn']
        elif key == 'damping':
            self.eta = kwargs['eta']
        elif key == 'hydrodynamics':
            self.rho = kwargs['rho']    # REUSING
            self.cd = kwargs['cd']
        elif key == 'aerodynamics':
            self.rho = kwargs['rho']    # REUSING
            self.cd = kwargs['cd']
        elif key == 'thrust':
            self.thrust_coeff = kwargs['thrust_coeff']
        elif key == 'pointForces':
            self.point_force_node_indices = kwargs['point_force_node_indices']
            self.point_force_vectors = kwargs['point_force_vectors']
        # TODO: Translate the following forces
        elif key == 'selfContact':
            self.delta = kwargs['delta']
            self.h = kwargs['h']
            self.imc_stiffness = kwargs['kc']
        elif key == 'shellContact':
            self.delta = kwargs['delta']
            self.h = kwargs['h']
            self.imc_stiffness = kwargs['kc']
        elif key == 'selfContactExtendedArc':
            self.beam_radius = kwargs['beam_radius']
            self.imc_stiffness = kwargs.get('kc', 1.0)
            self.self_contact_min_sep = kwargs.get('min_segment_separation', 6)
        elif key == 'selfFriction':
            self.mu = kwargs['mu']
            self.vel_tol = kwargs['vel_tol']
        elif key == 'floorContact':
            self.ground_z = kwargs['ground_z']
            self.ground_stiffness = kwargs['stiffness']
            self.ground_delta = kwargs['delta']
            self.ground_h = kwargs['h']
        elif key == 'floorFriction':
            self.ground_mu = kwargs['mu']
            self.ground_vel_tol = kwargs['vel_tol']
        elif key == 'beamToBeamContactExtendedArc':
            self.beam_radius = kwargs['beam_radius']
            self.imc_stiffness = kwargs.get('kc', 1.0)
            self.n_nodes_beam_A = kwargs['n_nodes_beam_A']
            self.d_imc = kwargs.get('d_imc', 0.008)
            self.kappa_threshold = kwargs.get('kappa_threshold', None)
        elif key == 'selfContactEberly3D':
            self.beam_radius = kwargs['beam_radius']
            self.imc_stiffness = kwargs.get('kc', 1.0)
            self.self_contact_min_sep = kwargs.get('min_segment_separation', 6)
            self.d_imc = kwargs.get('d_imc', 0.0008)
        elif key == 'beamToBeamContactEberly3D':
            self.beam_radius = kwargs['beam_radius']
            self.imc_stiffness = kwargs.get('kc', 1.0)
            self.n_nodes_beam_A = kwargs['n_nodes_beam_A']
            self.d_imc = kwargs.get('d_imc', 0.0008)
        elif key == 'selfFrictionEberly3D':
            # Coulomb friction layered on top of selfContactEberly3D.
            # Reuses mu / vel_tol attribute names so the Eberly and IMC
            # friction models share parameters when both are configured.
            self.mu = kwargs['mu']
            self.vel_tol = kwargs['vel_tol']
        elif key == 'beamToBeamFrictionEberly3D':
            # Coulomb friction layered on top of beamToBeamContactEberly3D.
            self.mu = kwargs['mu']
            self.vel_tol = kwargs['vel_tol']
        elif key == 'circleFourNodesIMCWeighted':
            # Parameters for 2D circle contact based on extended-arc IMC (Esther)
            # circle_center is a 2D np.array [x, z] in the simulation plane
            self.circle_center = kwargs['circle_center']
            self.circle_radius = kwargs['circle_radius']
            # Typically use rod radius as beam_radius
            self.beam_radius = kwargs['beam_radius']
            # IMC stiffness scaling (passed through to the contact law if needed)
            self.imc_stiffness = kwargs.get('kc', 1.0)
        elif key == 'circleFourNodesIMCWeightedXY':
            # Parameters for 2D circle contact in the (x,y) plane based on extended-arc IMC
            # circle_center is a 3D np.array [x, y, z] in world coordinates
            self.circle_center_3d = kwargs['circle_center']
            self.circle_radius = kwargs['circle_radius']
            self.beam_radius = kwargs['beam_radius']
            self.imc_stiffness = kwargs.get('kc', 1.0)
        elif key == 'circleNodeBead':
            # Simple bead-distance contact: each beam node is a sphere of
            # radius `bead_radius`, the obstacle is a sphere of radius
            # `circle_radius` centred at `circle_center` (3D).
            #
            # `bead_radius` is INDEPENDENT of the rod cross-section radius.
            # The natural choice is half the inter-node spacing:
            #     bead_radius = L / (2 (N - 1))
            # which makes adjacent beads tangent in the undeformed state.
            # If omitted we fall back to the rod radius `beam_radius` for
            # backwards compatibility.
            self.circle_center_3d = np.asarray(kwargs['circle_center'], dtype=float).reshape(3)
            self.circle_radius = kwargs['circle_radius']
            self.beam_radius = kwargs['beam_radius']
            self.bead_radius = kwargs.get('bead_radius', self.beam_radius)
            self.imc_stiffness = kwargs.get('kc', 1.0)
            self.d_imc = kwargs.get('d_imc', 0.0008)
            self.n_nodes_beam_A = kwargs.get('n_nodes_beam', None)
        else:
            raise KeyError
        
        self.__ext_force_list.append(key)

    @property
    def ext_force_list(self):
        return self.__ext_force_list

    def set_static(self):
        if hasattr(self, 'g'):
            self.static_g = self.g
