import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate

class Sir:

  def __init__(self, init_infect, filename, delimiter): #init infect = numero int di nodi infetti
    df = pd.read_csv('/content/'+filename, delimiter) #delimiter di default = ','
    self.graph = nx.from_pandas_edgelist(df, source='from', target='to')
    self.init_infect = init_infect
    self.init_state = self.set_init_infect()
  
  def set_init_infect(self):
    #get graph nodes
    nodes = self.graph.nodes
    sample = rd.sample(nodes, self.init_infect) #assegno tot nodi casuali come infetti
    
    #set parametri iniziali: infetti iniziali
    for node in nodes:   
      #set initial infected
      if node in sample:
        nodes[node]['State'] = 'I'
        nodes[node]['T_rec'] = 0
        nodes[node]['T_sus'] = 0
        nodes[node]['DNA'] = [node]
        nodes[node]['INIT'] = True
      else:
        nodes[node]['State'] = 'S'
        nodes[node]['T_rec'] = 0
        nodes[node]['T_sus'] = 0
        nodes[node]['DNA'] = []
        nodes[node]['INIT'] = False
    
    #self.draw_network()
    infected = self.getInfected()
    print(f"Nodi Tot. Grafo:{len(nodes)}")
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
  
  def whoInfects(self, node):
    infects = []
    for n in self.graph.nodes:
      for nd in self.graph.nodes[n]['DNA']:
        if nd == node:
          infects.append(n)
    
    return infects
  
  def spread_count(self): #to start after a simulation
    
    nodes = self.graph.nodes
    nodes_count = {}
    
    for n in nodes:
      for node in nodes[n]['DNA']:
        if node in nodes_count:
          nodes_count[node] += 1
        else:
          nodes_count[node] = 1
    
    df = pd.DataFrame.from_dict({
    'NODE': list(nodes_count.keys()),
    'SPREAD_COUNTS': list(nodes_count.values())
    })
    sort_counts = df.sort_values('SPREAD_COUNTS', ascending=False)
    sort_counts.index = np.arange(1, len(sort_counts) + 1)
    sort_counts.index.name = 'RANK'
    top_spreader = sort_counts[:15]
    print(tabulate(top_spreader, headers='keys', tablefmt='psql'))
    
    return top_spreader

  
  #method to start a single simulation
  def simulation(self, p_trans, t_rec, t_sus, t_sim):
    
    G = self.graph
    #Initial/live/final data list per grafici e statistiche
    tot_S = []
    tot_I = []
    tot_R = []
    Timeline = [] #array temporale per grafico e statistiche
    
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
            G.nodes[n]['DNA'].append(i)

      #guarigione
      for j in infected:
        if G.nodes[j]['T_rec'] >= t_rec:
          G.nodes[j]['State'] = 'R'
          G.nodes[j]['T_sus'] = 0
          G.nodes[j]['T_rec'] = 0
        else:
          G.nodes[j]['T_rec'] += 1
    
      #ritornare ad essere suscettibili
      for r in recovered:
        if G.nodes[r]['T_sus'] >= t_sus:
          G.nodes[r]['State'] = 'S'
          G.nodes[r]['T_rec'] = 0
          G.nodes[r]['T_sus'] = 0
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
    spreaders = self.spread_count()
    #self.draw_network()
    return spreaders
    
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
    
    plt.figure(figsize=(12,6))
    #pos = nx.spring_layout(g, iterations = 15, seed=1721)
    pos = nx.random_layout(self.graph, center=None, dim=2, seed=None)
    nx.draw_networkx_nodes(self.graph, pos, node_size=40, node_color=color_map)
    nx.draw_networkx_edges(self.graph, pos, alpha=0.2)
    plt.axis('off')
    plt.show() 
    #return tot_S, tot_I, tot_R
  
    