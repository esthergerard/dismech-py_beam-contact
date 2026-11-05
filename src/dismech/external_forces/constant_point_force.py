import numpy as np


def add_point_forces(robot, node_indices, force_vectors):
    fpt = np.zeros_like(robot.state.q)

    # Ensure numpy array
    force_vectors = np.asarray(force_vectors)

    # map_node_to_dof returns an (n, 3) index array for a node list; match
    # the force array to that shape (the old flat reshape crashed for n > 1)
    dof_indices = robot.map_node_to_dof(node_indices)
    fpt[dof_indices] = force_vectors.reshape(dof_indices.shape)
    # print(f"Adding point forces at nodes {node_indices} with force vectors {force_vectors}")
    return fpt
