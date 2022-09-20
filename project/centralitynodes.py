'''
Python module for compare the super spreaders node with the node with high value of centrality metrics
(Used in colab notebook)
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from tabulate import tabulate
import matplotlib.colors as mcolors
from networkx.algorithms.centrality.eigenvector import eigenvector_centrality
from networkx.algorithms.centrality.betweenness import betweenness_centrality
from networkx.algorithms.centrality.degree_alg import degree_centrality
from networkx.algorithms.bipartite.centrality import closeness_centrality

def getNetwork(filename, delimiter):
    df = pd.read_csv('/content/'+filename, delimiter) #delimiter di default = ','
    graph = nx.from_pandas_edgelist(df, source='from', target='to')
    
    return graph

def centrality_metrics(graph):
    dc = nx.degree_centrality(graph)
    cc = nx.closeness_centrality(graph)
    bc = nx.betweenness_centrality(graph)
    ec = nx.eigenvector_centrality(graph)
    vr = nx.voterank(graph, 15) #primi 15 più votati
    
    #from dict to pandas df
    dc_df = pd.DataFrame.from_dict({
    'node': list(dc.keys()),
    'degree centrality': list(dc.values())
    })

    cc_df = pd.DataFrame.from_dict({
    'node': list(cc.keys()),
    'closeness centrality': list(cc.values())
    })

    bc_df = pd.DataFrame.from_dict({
    'node': list(bc.keys()),
    'betweeness centrality': list(bc.values())
    })

    ec_df = pd.DataFrame.from_dict({
    'node': list(ec.keys()),
    'eigenvector centrality': list(ec.values())
    })
    
    vr_df = pd.DataFrame({'node':vr})

    return dc_df, cc_df, bc_df, ec_df, vr_df
  
def heatmap_centrality(G, metrics_name):
    
    cm = {}
    #get metrics
    if metrics_name == 'degree centrality':
      cm = nx.degree_centrality(G)
    if metrics_name == 'closeness centrality':
      cm = nx.closeness_centrality(G)
    if metrics_name == 'betweenness centrality':
      cm = nx.betweenness_centrality(G)
    if metrics_name == 'eigenvector centrality':
      cm = nx.eigenvector_centrality(G)
    
    df = pd.DataFrame.from_dict({
    'node': list(cm.keys()),
    metrics_name: list(cm.values())
    })
    sort_metrics = df.sort_values(metrics_name, ascending=False)
    top_metrics = sort_metrics[:15]

    plt.figure(figsize=(16,8))
    pos = nx.spring_layout(G, iterations= 15, seed=1721)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=40, cmap=plt.cm.plasma, 
                                  node_color=list(cm.values()),
                                  nodelist=cm.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    edges = nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    plt.title(metrics_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()
    print("Top 15:")
    print(top_metrics)
    

def top_centrality_node(graph):
    
    #get df centrality metrics
    dc, cc, bc, ec, VR = centrality_metrics(graph)
    #sort top 10 and combine dataframe 
    ec_sorted = ec.sort_values('eigenvector centrality', ascending=False)
    ec_sorted.index = np.arange(1, len(ec_sorted) + 1)

    cc_sorted = cc.sort_values('closeness centrality', ascending=False)
    cc_sorted.index = np.arange(1, len(cc_sorted) + 1)

    bc_sorted = bc.sort_values('betweeness centrality', ascending=False)
    bc_sorted.index = np.arange(1, len(bc_sorted) + 1)

    dc_sorted = dc.sort_values('degree centrality', ascending=False)
    dc_sorted.index = np.arange(1, len(dc_sorted) + 1)
    #voterank già ordina per importanza valore(rank), quindi setto solo gli index
    VR.index = np.arange(1, len(VR) + 1)
    
    BC = bc_sorted['node'][:15]
    CC = cc_sorted['node'][:15]
    EC = ec_sorted['node'][:15]
    DC = dc_sorted['node'][:15]

    ss =pd.merge(BC, CC, left_index=True, right_index=True)
    ss =pd.merge(ss, EC, left_index=True, right_index=True)
    ss =pd.merge(ss, DC, left_index=True, right_index=True)
    ss =pd.merge(ss, VR, left_index=True, right_index=True)
    ss.columns = ['BC','CC','EC','DC','VR']
    ss.index.name = 'RANK'
    print(tabulate(ss, headers='keys', tablefmt='psql'))