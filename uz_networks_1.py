#pip3 install networkx
#pip3 install scipy

import networkx as nx           # Network operations
import numpy as np              # Reading .mtx
from scipy.io import mmread     # Reading .mtx
import matplotlib.pyplot as plt        # Basic plotting
import operator                 # Sorting and converting dictionary to a list
import pandas as pd             # Tables and visualization
import collections

def mtx_to_graphi(graph, name):
    """Converts mtx file to graphi file for further visualization

    Args:
        graph (graph): .mtx file graph loaded in networkx.Graph() method.
        name (str):     output file name
    """

    with open('data/'+ name +'.graphml', 'wb') as output_file:
        nx.write_graphml(graph, output_file)
    
    print(name +'.graphml file generated')
        

def network_elements(graph):

    """Prints basic graph properties

    Args:
        graph (graph): .mtx file graph loaded in networkx.Graph() method.
    """

    # Check if network is directed
    print('Basic graph info: ')
    print('Is directed:', nx.is_directed(graph))
    print('Is weighted:', nx.is_weighted(graph))
    print('Is connected:',nx.is_connected(graph))

    
    # Assignment 1
    total_nodes = graph.number_of_nodes()
    print('Broj cvorova N u grafu: ' + str(total_nodes))
    total_edges = graph.number_of_edges()
    print('Broj veza K u grafu: ' + str(total_edges))
    print('Prosjecan broj ulaznih/izlaznih veza: ' + str(len(list(nx.average_degree_connectivity(graph)))))
    # Assignmnt 2
    # Not a directed graph, so average input/output connections are not calculated.
    
    # Assignment 3
    total_weight = graph.size(weight='weight')
    avg_weight = total_weight / total_nodes
    print('Ukupna snaga grafa: ' + str(total_weight))
    print('Prosjecna snaga grafa: ' + str(avg_weight))
    
    # Assignment 4
    conn_comp = nx.number_connected_components(graph)
    conn_comp_max = len(list(max(nx.connected_components(graph), key=len)))
    print('Broj komponenti grafa: ' + str(conn_comp))
    print('Velicina najvece komponente grafa: ' + str(conn_comp_max))
    
    # Assignment 5
    avg_path = nx.average_shortest_path_length(graph)
    print('Prosjecni najkraci put grafa: ' + str(avg_path))
    diam = nx.diameter(graph)
    print('Diametar grafa: ' + str(diam))
    # Eccentricity type: dictionary
    eccent = nx.eccentricity(graph)
    avg_eccent = float(sum(eccent.values())) / len(eccent)
    print('Prosjecna ekscentricnost grafa: ' + str(avg_eccent))
   
    # Assignment 6
    global_eff = nx.global_efficiency(graph)
    print('Globalna ucinkovitost: ' + str(global_eff))
  
    # Assignement 7
    glob_clustering = len(nx.clustering(graph))
    print('Globalni koeficijent grupiranja: ' + str(glob_clustering))
   
    # Assignment 8
    avg_clustering = nx.average_clustering(graph)
    print('Prosjecni koeficijent grupiranja: ' + str(avg_clustering))
    
    # Assignment 9
    node_assortativity = nx.degree_assortativity_coefficient(graph)
    print('Asortativnost s obzirom na stupanj cvora: ' + str(node_assortativity))
    
    # Assignment 10    

    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticklabels(deg)
    plt.savefig('plots/plot_degree.png')
    

    #ToDo
    #weights = [graph.degree(n, weight='weight') for n in graph.nodes()]
    #plt.hist(weights)
    
    # Assignment 11
    degree_cent = nx.degree_centrality(graph)
    betw_cent = nx.betweenness_centrality(graph)
    closeness_cent = nx.closeness_centrality(graph)

    # Converted dict to list
    sorted_degree = sorted(degree_cent.items(), key=operator.itemgetter(1), reverse=True)
    sorted_between = sorted(betw_cent.items(), key=operator.itemgetter(1), reverse=True)
    sorted_closeness = sorted(closeness_cent.items(), key=operator.itemgetter(1), reverse=True)
    
    degree_df = pd.DataFrame(sorted_degree, columns=['Node','Degree Centrality'])
    between_df = pd.DataFrame(sorted_between, columns=['Node','Betweeness Centrality'])
    closeness_df = pd.DataFrame(sorted_closeness, columns=['Node','Closeness Centrality'])
    
    print('Degree: ')
    print(degree_df.head(n=10))
    print('Betweeness: ')
    print(between_df.head(n=10))
    print('Closeness: ')
    print(closeness_df.head(n=10))
    
    # Assignment 12
    avg_closeness = float(sum(closeness_cent.values())) / len(closeness_cent)
    print('Prosjecna centralnost blizine: ' + str(avg_closeness))

    # Assignment 13
    avg_between = float(sum(betw_cent.values())) / len(betw_cent)
    print('Prosjecna medupolozenost: ' + str(avg_between))

    #partition = community_louvain.best_partition(graph)
    #print(partition)
    
    # Assignemnt 14
    
    #ValueError: Bad node degree (-12.068279800918697)
    
    


def centralities(graph):
    """Calculates centrality tables 
    Instructions: 
    https://networkx.org/documentation/stable/reference/algorithms/centrality.html

    Args:
        graph (graph): .mtx file graph loaded in networkx.Graph() method.
    """
    '''
    eigenvector_cent = nx.eigenvector_centrality(graph)
    print('Centrality 1 generated')
    harmonic_cent = nx.harmonic_centrality(graph)
    print('Centrality 2 generated')
    subgraph_cent = nx.subgraph_centrality(graph)
    print('Centrality 3 generated')
    curr_clos_cent = nx.current_flow_closeness_centrality(graph)
    print('Centrality 4 generated')
    '''
    curr_betw_cent = nx.load_centrality(graph)
    print('Centrality 5 generated')

    '''
    # Converting dict to list
    eigenvector_sorted = sorted(eigenvector_cent.items(), key=operator.itemgetter(1), reverse=True)
    harmonic_sorted =  sorted(harmonic_cent.items(), key=operator.itemgetter(1), reverse=True)
    subgraph_sorted =  sorted(subgraph_cent.items(), key=operator.itemgetter(1), reverse=True)
    curr_clos_sorted =  sorted(curr_clos_cent.items(), key=operator.itemgetter(1), reverse=True)
    '''
    curr_betw_sorted =  sorted(curr_betw_cent.items(), key=operator.itemgetter(1), reverse=True)
    '''
    # Converting list to dataframe
    eigenvector_df = pd.DataFrame(eigenvector_sorted, columns=['Node','Eigen Vector'])
    harmonic_df = pd.DataFrame(harmonic_sorted, columns=['Node','Harmonic'])
    subgraph_df = pd.DataFrame(subgraph_sorted, columns=['Node','Subgraph'])
    curr_clos_df = pd.DataFrame(curr_clos_sorted, columns=['Node','Current Flow closeness'])
    '''
    curr_betw_df = pd.DataFrame(curr_betw_sorted, columns=['Node','Current Flow betweeness'])
    '''
    # Printing tables
    print('\n5 ADDITIONAL CENTRALITIES:\n')
    print('Eigenvector: ')
    print(eigenvector_df.head(n=10))
    print('Harmonic: ')
    print(harmonic_df.head(n=10))
    print('Subgraph: ')
    print(subgraph_df.head(n=10))
    print('Current flow closeness: ')
    print(curr_clos_df.head(n=10))
    '''
    print('Current flow betweeness: ')
    print(curr_betw_df.head(n=10))
    

# Data preparation and fetching
#mahindas = mmread('data/econ-mahindas.mtx')
#G_mahindas = nx.Graph(mahindas)
#mtx_to_graphi(G_mahindas)
#network_elements(G_mahindas)
#centralities(G_mahindas)

# https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/psmigr/psmigr.html
psmigr1 = mmread('data/econ-psmigr1.mtx')
G_psmigr1 = nx.Graph(psmigr1)

#mtx_to_graphi(G_psmigr1, 'psmigr1')
#network_elements(G_psmigr1)
centralities(G_psmigr1)

# Function calls

