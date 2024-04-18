import numpy as np

xyz = np.genfromtxt("geom_opt.xyz", delimiter=" ", skip_header=2, dtype=None)

def distance_matrix(xyz):
    """Construct an upper triangle matrix where the distance between each particle is computed.
    The diagonal is the distance between itself and should remain 0."""
    
    #Construct an empty square matrix where the shape matches the amount of particles N in the system 
    N = len(xyz)
    distance_matrix = np.zeros((N,N))
    
    #For every particle, loop over every neighbor and compute the distance
    for atom in range(N):
        x1, y1, z1 = xyz[atom][1], xyz[atom][2], xyz[atom][3]
        for neighbor in range(atom, N):
            x2, y2, z2 = xyz[neighbor][1], xyz[neighbor][2], xyz[neighbor][3]
            distance_matrix[atom,neighbor] = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            
    return distance_matrix


def neighbor_shell(xyz, cutoff={"Si":2.6, "O":2.0}):
    """Takes the coordinates of a system and a dictionary that includes the cutoff per element and 
    returns an 1D array that corresponds to the amount of neighboring atoms within the cutoff"""
    
    #Construct a preliminary array that has the same length as the amount of atoms in the system
    N = len(xyz)
    neighbor_shell = np.zeros(N) 
    
    #Mirror the distance matrix to make life easier in the following steps
    dist_matrix = distance_matrix(xyz)
    dist_matrix += dist_matrix.T
    
    #Loop over all atoms and all of their neighbors and check which neighbors fall within the custom cutoff
    for atom in range(N):
        for neighbor in dist_matrix[atom]:
            if neighbor <= cutoff[xyz[atom][0].astype(str)]:
                neighbor_shell[atom] += 1 
    return neighbor_shell