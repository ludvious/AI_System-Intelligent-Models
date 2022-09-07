from email import iterators
import pandas as pd
from tabulate import tabulate
import numpy as np
import csv
import random as rd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from networkx.algorithms.centrality.eigenvector import eigenvector_centrality
from networkx.algorithms.centrality.betweenness import betweenness_centrality
from networkx.algorithms.centrality.degree_alg import degree_centrality
from networkx.algorithms.bipartite.centrality import closeness_centrality
import matplotlib.colors as mcolors

class Sir:

  def __init__(self, init_infect, filename, delimiter=' '): #init infect = numero int di nodi infetti
    df = pd.read_csv('/content/'+filename, delimiter) #delimiter di default = ','
    self.graph = nx.from_pandas_edgelist(df, source='from', target='to')
    self.init_infect = init_infect
    self.init_state = self.set_init_infect()
  
  def set_init_infect(self):
    #get nodes
    nodes = self.graph.nodes()
    sample = rd.sample(nodes, self.init_infect) #assegno tot nodi casuali come infetti
    color_map = []
    #COLORI STATE
    CELESTE = '#abcdef'
    ROSSO = '#ff5349'
    #set parametri iniziali
    for node in nodes:   
      #set initial infected
      if node in sample:
        self.graph.nodes[node]['State'] = 'I'
        self.graph.nodes[node]['T_rec'] = 0
        self.graph.nodes[node]['T_sus'] = 0
        color_map.append(ROSSO) #colore rosso
      else:
        self.graph.nodes[node]['State'] = 'S'
        self.graph.nodes[node]['T_rec'] = 0
        self.graph.nodes[node]['T_sus'] = 0
        color_map.append(CELESTE) # colore celeste
    
    self.draw_network()
    infected = self.getInfected()
    print(f"Nodi Tot. Grafo:{len(self.graph.nodes)}")
    print(f"Nodi Infetti Iniziali:{len(infected)}")
    print(f":{self.getInfected()}")
  
  def getNeighbors(self, node):
    return [n for n in self.graph.neighbors(node)]

  def getNeighbors_Susceptibile(self, node):
    neigh_susceptible = []
    for n in self.graph.neighbors(node):
        if self.graph.nodes[n]['State'] == 'S':
            neigh_susceptible.append(n)
    
    return neigh_susceptible

  def getSusceptible(self):
    susceptible = []
    for n in self.graph.nodes:
        if self.graph.nodes[n]['State'] == 'S':
            susceptible.append(n)

    return susceptible

  def getInfected(self):
    infected = []
    for n in self.graph.nodes:
        if self.graph.nodes[n]['State'] == 'I':
            infected.append(n)
  
    return infected

  def getRecovered(self):
    recovered = []
    for n in self.graph.nodes:
        if self.graph.nodes[n]['State'] == 'R':
            recovered.append(n)

    return recovered
  
  def simulation(self, p_trans, t_rec, t_sus, t_sim):
    
    G = self.graph
    #Initial/live/final data list per grafici e statistiche
    tot_S = []
    tot_I = []
    tot_R = []
    Timeline = [] #array temporale per grafico e statistiche
    color_map = []
    
    for t in range(1,t_sim+1):
      Timeline.append(t)
      #get nodes
      susceptible = self.getSusceptible()
      infected = self.getInfected()
      recovered = self.getRecovered()

      for i in infected:
        neighbors = self.getNeighbors_Susceptibile(i)
        for n in neighbors:
          #set random probablilty for that node
          p = np.random.rand()
          if p <= p_trans:
            G.nodes[n]['State'] = 'I'
            G.nodes[n]['T_rec'] = 0
            G.nodes[n]['T_sus'] = 0
            #color_map.append('#ff5349') #colore rosso

      #guarigione
      for j in infected:
        if G.nodes[j]['T_rec'] >= t_rec:
          G.nodes[j]['State'] = 'R'
          G.nodes[j]['T_sus'] = 0
          G.nodes[j]['T_rec'] = 0
          #color_map.append('#98ff98') #colore verde
        else:
          G.nodes[j]['T_rec'] += 1
    
      #ritornare ad essere suscettibili
      for r in recovered:
        if G.nodes[r]['T_sus'] >= t_sus:
          G.nodes[r]['State'] = 'S'
          G.nodes[r]['T_rec'] = 0
          G.nodes[r]['T_sus'] = 0
          #color_map.append('#abcdef') # colore celeste
        else:
          G.nodes[r]['T_sus'] += 1
      
      #update
      susceptible = self.getSusceptible()
      infected = self.getInfected()
      recovered = self.getRecovered()

      s_upd = len(susceptible)
      i_upd = len(infected)
      r_upd = len(recovered)

      tot_S.append(len(susceptible))
      tot_I.append(len(infected))
      tot_R.append(len(recovered))

    
    plt.plot(Timeline, tot_S, label="SIR Susceptible")
    plt.plot(Timeline, tot_I, label="SIR Infected")
    plt.plot(Timeline, tot_R, label="SIR Recovered")
    plt.title('SIR Simulation')
    plt.show()
    print(f"Giorno {t} :\nSuscettibili: {s_upd}\nInfetti: {i_upd}\nGuariti: {r_upd}")
    self.draw_network()
    #return tot_S, tot_I, tot_R
    
  #draw the network with node states
  def draw_network(self):
    color_map = []
    susceptible = self.getSusceptible()
    infected = self.getInfected()
    recovered = self.getRecovered()

    for s in susceptible:
      color_map.append('#abcdef') # colore celeste

    for i in infected:
      color_map.append('#ff5349') #colore rosso
    
    for r in recovered:
      color_map.append('#98ff98') #colore verde
    
    plt.figure(figsize=(20,10))
    #pos = nx.spring_layout(g, iterations = 15, seed=1721)
    pos = nx.random_layout(self.graph, center=None, dim=2, seed=None)
    nx.draw_networkx_nodes(self.graph, pos, node_size=40, node_color=color_map)
    nx.draw_networkx_edges(self.graph, pos, alpha=0.2)
    plt.axis('off')
    plt.show() 

    #return tot_S, tot_I, tot_R
  
  def centrality_metrics(self):
    
    G = self.graph
    dc = nx.degree_centrality(G)
    cc = nx.closeness_centrality(G)
    bc = nx.betweenness_centrality(G)
    ec = nx.eigenvector_centrality(G)
    vr = nx.voterank(G, 15) #primi 15 più votati
    
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
  
  def heatmap_centrality(self, metrics_name):
    
    G = self.graph
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
    
  def super_spreader(self):
    
    #get df centrality metrics
    dc, cc, bc, ec, VR = self.centrality_metrics()
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
    print(tabulate(ss, headers='keys', tablefmt='psql'))