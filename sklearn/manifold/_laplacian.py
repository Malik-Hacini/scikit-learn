import numpy as np


class Laplacian:
    """
    A class to compute the possible Laplacian matrices of a directed graph.
    Based on the "Generalized Spectral Clustering" paper at  	
    https://doi.org/10.48550/arXiv.2203.03221 (link to update)
    The so-called "generalized Laplacians" parameters can be tuned to coincide with the classical unidrected Laplacians."""

    def __init__(self, adjacency_matrix, standard=False, measure=(3,0.7,0.9)):
        """
        Initializes the Laplacian class with the adjacency matrix.
        
        Args:
            adjacency_matrix (ndarray): The adjacency matrix of the graph.
            standard (bool): If True, uses the standard Laplacian measure (1, 1, 1). Defaults to False.
            measure (tuple[float, float, float]) : The parameters of the vertex measure used for the Laplacians, respectively (t, alpha, gamma).   
        """
        self.adjacency_matrix = adjacency_matrix
        self.N = adjacency_matrix.shape[0]  # Number of nodes in the graph
        if standard:
            self.measure = (1,1,1) # Measure defining the standard Laplacians.
        else:
            self.measure = measure
        
        """Compute the matrices necessary for all generalized Laplacians."""
        t,alpha,gamma = self.measure
        D = np.diag(np.sum(self.adjacency_matrix, axis=1))
        P = np.linalg.inv(D) @ self.adjacency_matrix
        P_gamma=gamma*P + ((1-gamma)*1/self.N)*np.ones((self.N,self.N))
        v=(((1/self.N)*np.matmul(np.ones((1,self.N)),np.linalg.matrix_power(P_gamma,t)))**alpha).transpose()
        D_v = np.diag(v.flatten())  
        xi = P.transpose() @ v
        D_xi = np.diag(xi.flatten())

        self.natural_transition_matrix = P
        self.measure_xi_matrix = D_xi
        self.measure_transition_matrix = D_v
        print("P", P, "\nD_v", D_v, "\nD_xi", D_xi)
   
    def D_v_xi(self):
        """
        Computes the measure transition matrix D_{v + xi}.

        Returns:
            ndarray: The measure transition matrix.
        """
        D_v = self.measure_transition_matrix
        D_xi = self.measure_xi_matrix
        print("+", D_v + D_xi)
        return D_v + D_xi   
     
    def inv_sqrt_D_v_xi(self):
        """
        Computes the inverse square root of the measure transition matrix D_{v + xi}.

        Returns:
            ndarray: The inverse square root of the measure transition matrix.
        """
        return np.diag(1 / np.sqrt(np.diag(self.D_v_xi())))
    
    def unnormalized(self):
        """
        Computes the unnormalized generalized Laplacian matrix L_v = D_{v + xi} - (D_v * P + P^T * D_v)

        Returns:
            tuple(ndarray): (L_v, None)
        """
        P = self.natural_transition_matrix
        D_v = self.natural_transition_matrix
        L_v = self.D_v_xi() - (D_v @ P + P.T @ D_v)
        print("L_v", L_v)
        return L_v, np.diag(L_v) 
    
    def normalized(self):
        """
        Computes the normalized generalized Laplacian matrix L_norm_v = D_{v+xi}^(-1/2)   * L_{v} * D_{v+xi}^(-1/2).

        Returns:
            tuple(ndarray): (L_norm_v , None)
        """
        L_v = self.unnormalized()[0]
        inv_sqrt_D_v_xi = self.inv_sqrt_D_v_xi()
        print("inv_sqrt_D_v_xi", inv_sqrt_D_v_xi.shape)
        L_norm_v = inv_sqrt_D_v_xi @ L_v @ inv_sqrt_D_v_xi
        return L_norm_v, np.sqrt(np.diag(self.D_v_xi()))

    def random_walk(self):
        """
        Computes the random walk generalized Laplacian matrix L_rw_v = D_{v+xi}^(-1/2)*L_v 

        Returns:
            tuple(ndarray): (L, D)
        """
        L_v = self.unnormalized()[0]
        L_rw_v = self.inv_sqrt_D_v_xi() @ L_v
        return L_rw_v, np.diag(L_rw_v)


    