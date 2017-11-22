# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:16:11 2017

@author: muriel
"""
import networkx as nx
import community
import numpy as np
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import  squareform


def compute_edge_similarity_matrix(parcel_infidelity_matrix, significant_edges_matrix):
    
    """
    Computes and thresholds the edge similarity matrix 

    The matrix is then used as an input for the clustering.

    Parameters
    ----------
    parcel_infidelity_matrix : nb_parcesl x nb_parcels symmetric matrix
        Contains the parcel to parcel infidelity for each parcel-pair
    significant_edges_matrix : nb_parcesl x nb_parcels symmetric matrix    
        Thresholded adjacency matrix where significant edges are different 
        from 0.


    Returns
    -------
    edge_similarity_matrix: nb_edges x nb_edges matrix
        For each edge pair contains the similarity between edges with 
        similarity computed as the correlation between each edge's infidelity
        profile
    thresholded_similarity_matrix: nb_edges x nb_edges matrix
        Thresholded version of the edge_similarity_matrix in which the
        similarity values are set to zero if the correlation coeff p-value
        is inferior to alpha with alpha = 0.05
   """
   
       # Check that significant edges matrix is symmetric
    if not(np.allclose(significant_edges_matrix.transpose(), significant_edges_matrix)):
            raise RuntimeError('significant_edges_matrix'
                       ' should be symmetrical" ')
 
    # Check that significant edges matrix is symmetric
    if not(np.allclose(parcel_infidelity_matrix.transpose(), parcel_infidelity_matrix)):
            raise RuntimeError('parcel_infidelity_matrix'
                       ' should be symmetrical" ')
       
        # Identify, list and count significant edges 
    # symetric so we only look at upper triangle
    significant_edges_matrix *= np.tri(*significant_edges_matrix.shape)
    
    # List by row and column of significant edges
    sig_edges_list = np.transpose(np.nonzero(significant_edges_matrix)).astype(int)
    n_edges =  sig_edges_list.shape[0] 
  

    # Compute the edge-adjacency matrix only for significant edges 
     
    edge_adjacency_matrix = np.zeros([n_edges,n_edges])
    
    # Compute and fill upper triangle of symmetric edge_adjacency_matrix
    for i_edge_row in range(n_edges):
        for i_edge_column in range(i_edge_row + 1, n_edges):
    #        print([i_edge_row,i_edge_column])
            if (sig_edges_list[i_edge_row,0] != sig_edges_list[i_edge_column,0]  and 
                sig_edges_list[i_edge_row,0] != sig_edges_list[i_edge_column,1]  and
                sig_edges_list[i_edge_row,1] != sig_edges_list[i_edge_column,0]  and 
                sig_edges_list[i_edge_row,1] != sig_edges_list[i_edge_column,1] ):
                edge_adjacency_matrix[i_edge_row,i_edge_column] = max([
                       parcel_infidelity_matrix[sig_edges_list[i_edge_row,0], sig_edges_list[i_edge_column,0]]*
                       parcel_infidelity_matrix[sig_edges_list[i_edge_row,1], sig_edges_list[i_edge_column,1]],
                       parcel_infidelity_matrix[sig_edges_list[i_edge_row,0], sig_edges_list[i_edge_column,1]]*
                       parcel_infidelity_matrix[sig_edges_list[i_edge_row,1], sig_edges_list[i_edge_column,0]]])
            elif sig_edges_list[i_edge_row,0] == sig_edges_list[i_edge_column,0]:
                edge_adjacency_matrix[i_edge_row,i_edge_column] = parcel_infidelity_matrix[sig_edges_list[i_edge_row,1], sig_edges_list[i_edge_column,1]]**2
            elif sig_edges_list[i_edge_row,1] == sig_edges_list[i_edge_column,1]:
                edge_adjacency_matrix[i_edge_row,i_edge_column] = parcel_infidelity_matrix[sig_edges_list[i_edge_row,0], sig_edges_list[i_edge_column,0]]**2
            elif sig_edges_list[i_edge_row,0] == sig_edges_list[i_edge_column,1]:
                edge_adjacency_matrix[i_edge_row,i_edge_column] = parcel_infidelity_matrix[sig_edges_list[i_edge_row,1], sig_edges_list[i_edge_column,0]]**2
            elif sig_edges_list[i_edge_row,1] == sig_edges_list[i_edge_column,0]:
                edge_adjacency_matrix[i_edge_row,i_edge_column] = parcel_infidelity_matrix[sig_edges_list[i_edge_row,0], sig_edges_list[i_edge_column,1]]**2


    # Assign upper traingle to lower triangle to symmetrize edge_adjacency_matrix 
    edge_adjacency_matrix += np.transpose(edge_adjacency_matrix)    
        
    # Create similarity matrix by correlating rows of each pair of edges
    # Matrix is symmetrical, so we only fill top triangle
    edge_similarity_matrix = np.zeros([n_edges,n_edges])
    p_value_matrix = np.zeros([n_edges,n_edges])
    
    for i_row in range(n_edges):
        for i_col in range(i_row,n_edges):
            edge_similarity_matrix[i_row,i_col] = pearsonr(edge_adjacency_matrix[i_row,:], edge_adjacency_matrix[i_col,:])[0]
            p_value_matrix[i_row,i_col] = pearsonr(edge_adjacency_matrix[i_row,:], edge_adjacency_matrix[i_col,:])[1]
    
    # Assign upper triangles to lower triangles to symmetrize matrices 
    edge_similarity_matrix += np.transpose(np.triu(edge_similarity_matrix, k=1))
    p_value_matrix += np.transpose(np.triu(p_value_matrix, k=1))
    
    # remove negative correlations 
    thresholded_similarity_matrix = np.multiply(edge_similarity_matrix,
                                                 np.where(edge_similarity_matrix<0,0,1))
    
    # remove non-significant correlations
    thresholded_similarity_matrix = abs(np.multiply(thresholded_similarity_matrix,
                                                 np.where(p_value_matrix>=0.05,0,1)))

    return  edge_similarity_matrix, thresholded_similarity_matrix   
            


def cluster_edges_UPGMA(edge_similarity_matrix, nb_clusters):

    """
    Determines edge-clusters given an edge similarity matrix (thresholded or 
    not) and a max number of clusters using UPGMA hierarchical clustering.

    Parameters
    ----------
    edge_similarity_matrix: nb_edges x nb_edges matrix
        For each edge pair contains the similarity between edges with 
        similarity computed as the correlation between each edge's infidelity
        profile.
        The matrix can be thresholded (values under a given threshold set to 0)
        or not.

    nb_clusters : int
        The final number of clusters


    Returns
    -------
    UGMPA_clustered_tree: tree instance of the sci.Py clustering algorithm

    cluster_assignements: a nb_edges sized array which indicates the cluster 
    assignment for each edge for the given nb_clusters
    
    cluster_plot: a seaborn clustermap object
        
    """
    
        # Check that edge_similarity_matrix is symmetric
    if not(np.allclose(edge_similarity_matrix.transpose(), edge_similarity_matrix)):
            raise RuntimeError('edge_similarity_matrix'
                       ' should be symmetrical" ')
    
            
    # Check that nb_clusters is positive 
    if nb_clusters <= 0 :
            raise RuntimeError('nb_clusters should be a non-zero positive integer')
            
    # Check that nb_clusters is positive and inferior to the number of edges.
    if nb_clusters > edge_similarity_matrix.shape[0]:
            raise RuntimeError('nb_clusters should be inferior or equal to'
                       ' the number of significant edges." ')        

    
        
    # Convert thresholded_similarity_matrix to distance vector
    distance_vector = squareform(1-edge_similarity_matrix)

    # perform linkage over the distance matrix
    UGMPA_clustered_tree = linkage(distance_vector, 'average')
    
    cluster_assignements = fcluster(UGMPA_clustered_tree, nb_clusters, criterion='maxclust')
    
        
    return UGMPA_clustered_tree, cluster_assignements


def cluster_edges_Louvain(edge_similarity_matrix, Louvain_resolution):

    """
    Determines edge-clusters given an edge similarity matrix (thresholded or 
    not) and a resolution using the Louvain graph community detection 
    algorithm.

    Parameters
    ----------
    edge_similarity_matrix: nb_edges x nb_edges matrix
        For each edge pair contains the similarity between edges with 
        similarity computed as the correlation between each edge's infidelity
        profile.
        The matrix can be thresholded (values under a given threshold set to 0)
        or not.

    Louvain_resolution : float
        the resolution parameter for Louvain community detection

    Returns
    -------
    community_assignment: a nb_edges*1 sized array 
        indicates the community 
        assignment for each edge for the given resolution parameter
    edge_dendogram: a list of dictionnaries
        a tree where each level is a partition of the graph. 
        Level 0 has the smallest communities and Level len(dendogram) - 1 
        has the best partition
    """
    
    # Check that resolution parameter is givan for the Louvain condition
    if (Louvain_resolution <= 0 ):
        raise RuntimeError('Please provide a non-zero'
                       'positive value for the Louvain_resolution parameter')
    
    # Check that edge_similarity_matrix is symmetric
    if not(np.allclose(edge_similarity_matrix.transpose(), edge_similarity_matrix)):
            raise RuntimeError('edge_similarity_matrix'
                       ' should be symmetrical" ')    
    
    nb_edges = edge_similarity_matrix.shape[0]
    
    # Create graph from thresholded_similarity_matrix
    edge_similarity_graph = nx.Graph(abs(edge_similarity_matrix))
    
    #Create Louvain partition
    partition = community.best_partition(edge_similarity_graph, resolution = Louvain_resolution)
    edge_dendogram = community.generate_dendrogram(edge_similarity_graph, resolution  = Louvain_resolution)
    
   
    community_assignment = np.zeros(nb_edges)

    for i_edge in range(nb_edges):
        community_assignment[i_edge] = partition[i_edge]
    
    return community_assignment, edge_dendogram
    
  
def sort_edges_by_cluster_size(cluster_assignments, edge_similarity_matrix):
    
    """
    Sorts the edges of an edge_similarity_matrix in order of increasing
    cluster size for a given cluster_assignement vector.

    Parameters
    ----------
        cluster_assignments: a nb_edges*1 sized array 
        indicates the cluster assignment for each edge.
        
        edge_similarity_matrix: nb_edges x nb_edges matrix
        For each edge pair contains the similarity between edges with 
        similarity computed as the correlation between each edge's infidelity
        profile.
        The matrix can be thresholded (values under a given threshold set to 0)
        or not.


    Returns
    -------
    cluster_sizes: a nb_clusters*3 sized array 
        Col 0 = cluster number assigned in the cluster_assignments vector
        Col 1 = cluster size (nb of edges in cluster)
        Col 2 = Cluster number when clusters ordered by increasing cluster size
        assignment for each edge for the given resolution parameter
    
    sorted_edge_matrix: nb_edges x nb_edges matrix
        edge_similarity_matrix with edges sorted by order of increasing
        cluster size 
    """
    nb_edges = cluster_assignments.shape[0]
    
    # If cluster numbering starts at 0, add one to each cluster number
    if min(cluster_assignments) == 0:
        cluster_assignments+=1
    
    cluster_assignments = cluster_assignments.astype(int)
    
    # Reorder matrix according to cluster assigmement
    # With largest clusters first.
    cluster_sizes = np.unique(cluster_assignments,return_counts = True)
    cluster_sizes = np.transpose(np.vstack((cluster_sizes[0], cluster_sizes[1])))
    
    nb_clusters = cluster_sizes.shape[0]
    
    # Add column indexing size order of each cluster
    cluster_sizes = np.append(cluster_sizes[np.argsort(cluster_sizes[:, 1])],
                                            np.transpose(range(nb_clusters)).reshape(nb_clusters,1),
                                            axis = 1)
    
    #Reorder array accordig to initial cluster order
    cluster_sizes = cluster_sizes[np.argsort(cluster_sizes[:, 0])]
    
    edges_ordered_by_cluster_size = np.zeros(nb_edges)
   
    # Assign to each edge a new cluster number according to cluster size
    for i_edge in range(nb_edges):
        edges_ordered_by_cluster_size[i_edge]=cluster_sizes[cluster_assignments[i_edge]-1, 2]
        
    edges_ordered_by_cluster_size = np.stack((edges_ordered_by_cluster_size,
                                              range(nb_edges)),
                                              axis = 1)
        
    edges_ordered_by_cluster_size = edges_ordered_by_cluster_size[np.argsort(edges_ordered_by_cluster_size[:,0])].astype(int)   
        
    sorted_edge_matrix = np.zeros([nb_edges,nb_edges])
    
    # Loop across all edges and assign new cluster nb according to cluster size
    for i_edge_row in range(nb_edges):
        for i_edge_col in range( i_edge_row + 1, nb_edges):
            sorted_edge_matrix[i_edge_row,i_edge_col] = edge_similarity_matrix[edges_ordered_by_cluster_size[i_edge_row,1],
                               edges_ordered_by_cluster_size[i_edge_col,1]]
            
    # Assign upper traingle to lower triangle to symmetrize edge_adjacency_matrix 
    sorted_edge_matrix += np.transpose(sorted_edge_matrix) 
    
    return cluster_sizes, sorted_edge_matrix
    
    