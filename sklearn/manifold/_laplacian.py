import numpy as np
from scipy.sparse import issparse, diags_array


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
        print("Adjacency Matrix", adjacency)
        self.N = adjacency.shape[0]  # Number of nodes in the graph
        self.measure=measure
        """Compute the matrices necessary for all generalized Laplacians."""
        self.is_sparse = issparse(self.adjacency)
        
        if self.is_sparse:
            degree_vec = self.adjacency.sum(axis=1).A1
        else:
            degree_vec = self.adjacency.sum(axis=1)
        P = self.adjacency / degree_vec

        if standard:
            v = degree_vec / degree_vec.sum()
            xi = np.zeros(self.N)
        else:
            t,alpha,gamma = measure
            P_gamma_t = self._get_P_gamma(P, gamma, t)
            v = (((1/self.N) * np.ones((1, self.N)) @ P_gamma_t)**alpha).T.flatten()   
            xi = P.T @ v

        self.natural_transition_matrix = P
        self.v_vector = v
        self.xi_vector = xi
        self.v_xi_sum = self.v_vector + self.xi_vector
   
    
    def unnormalized(self):
        """
        Computes the unnormalized generalized Laplacian matrix L_v = D_{v + xi} - (D_v * P + P^T * D_v)

        Returns:
            tuple: (L_v, diagonal of L_v)
        """
        P = self.natural_transition_matrix
        
        
        if self.is_sparse:
            D_v_xi = diags_array(self.v_xi_sum)
            D_v = diags_array(self.v_vector)
            
        else:
            D_v_xi = np.diag(self.v_xi_sum)
            D_v = np.diag(self.v_vector)
        
        L_v = D_v_xi - (D_v @ P + P.T @ D_v)
        diag = L_v.diagonal() if self.is_sparse else np.diag(L_v)
        
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
    
    def _get_P_gamma(self, P, gamma, t):
        """
         Compute P_gamma and its power for both sparse and dense cases.
        
        Args:
            P: Transition matrix (sparse or dense)
            gamma: Gamma parameter
            t: Power parameter
            
        Returns:
            P_gamma^t result as dense matrix
        """
        
        if self.is_sparse:
                P_dense = P.toarray()
        else:
                P_dense = P
            
        P_gamma = gamma * P_dense + ((1-gamma) / self.N) * np.ones((self.N, self.N))
            
        if t == 1:
            return P_gamma
        return np.linalg.matrix_power(P_gamma, t)




