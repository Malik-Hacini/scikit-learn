import numpy as np
from scipy.sparse import issparse, diags_array
from scipy.sparse.linalg import matrix_power


class Laplacian:
    """
    A class to compute the possible Laplacian matrices of a directed graph.
    Based on the "Generalized Spectral Clustering" paper at  	
    https://doi.org/10.48550/arXiv.2203.03221 (link to update)
    The so-called "generalized Laplacians" parameters can be tuned to coincide with the classical undirected Laplacians."""

    def __init__(self, adjacency, standard=False, measure=(3,0.7,1)):
        """
        Initializes the Laplacian class with the adjacency matrix.
        
        Args:
            adjacency (ndarray): The adjacency matrix of the graph.
            standard (bool): If True, uses the standard Laplacian measure (1, 1, 1). Defaults to False.
            measure (tuple[float, float, float]) : The parameters of the vertex measure used for the Laplacians, respectively (t, alpha, gamma).   
        """
        self.adjacency = adjacency
        self.N = adjacency.shape[0]  # Number of nodes in the graph
        self.measure=measure
        self.standard = standard
        """Compute the matrices necessary for all generalized Laplacians."""
        self.is_sparse = issparse(self.adjacency)
        
        degree_vec = self.adjacency.sum(axis=1).A1 if self.is_sparse else self.adjacency.sum(axis=1)
        P = self.adjacency / degree_vec
        if self.standard:
            v = degree_vec
            xi = np.zeros(self.N)
        else:
            t,alpha,gamma = measure
            if gamma==1:
                P_gamma_t = matrix_power(P.copy(), t)
            else:
                P_gamma_t = np.linalg.matrix_power((1-gamma)* np.full((self.N, self.N), 1/self.N) + gamma * P, t)
            v = (np.array(((1/self.N) * np.ones((1, self.N)) @ P_gamma_t)).T.flatten())**alpha
            xi = P.T @ v

        self.natural_transition_matrix = P
        self.v_vector = v
        self.xi_vector = xi
        self.v_xi_sum = self.v_vector + self.xi_vector
   
    
    def unnormalized(self):
        """
        Computes the unnormalized generalized Laplacian matrix L_v = D_{v + xi} - 1/2 * (D_v * P + P^T * D_v)

        Returns:
            tuple: (L_v, diagonal of L_v)
        """

        if self.is_sparse:
            D_v = diags_array(self.v_vector)
            if self.standard:
                L_v = D_v - self.adjacency
            else:
                D_v_xi = diags_array(self.v_xi_sum)
                D_times_P = D_v @ self.natural_transition_matrix
                L_v = D_v_xi - (D_times_P + D_times_P.T)
            
            diag = L_v.diagonal()
    
        
        else:
            if self.standard:
                L_v = self.adjacency * -1
                np.fill_diagonal(L_v, self.v_vector + np.diag(L_v))
            else:
                P = self.natural_transition_matrix
                D_times_P = self.v_vector[:, np.newaxis] * P
                L_v = (D_times_P + D_times_P.T)* -1
                np.fill_diagonal(L_v, self.v_xi_sum + np.diag(L_v))
            
            diag = np.diag(L_v)

        return L_v, diag
    
    def normalized(self):
        """
        Computes the normalized generalized Laplacian matrix L_norm_v = D_{v+xi}^(-1/2) * L_{v} * D_{v+xi}^(-1/2).

        Returns:
            tuple: (L_norm_v, sqrt(diagonal of D_{v+xi}))
        """
        L_v, _ = self.unnormalized()
        sqrt_v_xi_sum = np.sqrt(self.v_xi_sum)
        L_norm_v = L_v / (sqrt_v_xi_sum[:, np.newaxis] * sqrt_v_xi_sum[np.newaxis, :])
            
        return L_norm_v, sqrt_v_xi_sum

    def random_walk(self):
        """
        Computes the random walk generalized Laplacian matrix L_rw_v = D_{v+xi}^(-1)*L_v 

        Returns:
            tuple: (L_rw_v, diagonal of L_rw_v)
        """
        L_v, _ = self.unnormalized()
        L_rw_v = L_v / self.v_xi_sum[:, np.newaxis]     
        diag = L_rw_v.diagonal() if self.is_sparse else np.diag(L_rw_v)
            
        return L_rw_v, diag