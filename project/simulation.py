from project.NetworkSIR import Sir
from project import centralitynodes
import pandas as pd
import numpy as np
from tabulate import tabulate

# run random simulation and collect the node super spreaders for all simulation
def random_simulation(init_infect, filename, delimiter, p_trans, t_rec, t_sus, t_sim, num_sim):

  networks = []
  spreaders_df = pd.DataFrame()
  
  if num_sim > 1:
    for i in range(num_sim):
      networks.append(Sir(init_infect, filename, delimiter))
  
    for network in networks:
      spread_count = network.simulation(p_trans, t_rec, t_sus, t_sim)
      spreaders_df = spreaders_df.append(spread_count)
  
    top_spreaders = spreaders_df.groupby(['NODE'])['SPREAD_COUNTS'].count().reset_index(
    name='FREQUENCY').sort_values(['FREQUENCY'], ascending=False)
  
    top_spreaders.index = np.arange(1, len(top_spreaders) + 1)
    top_spreaders.index.name = 'RANK'
    top_15 = top_spreaders[:15]
    print(tabulate(top_15, headers='keys', tablefmt='psql'))
  else: #else if is only 1 sim, we get the draw graph output
    sim = Sir(init_infect, filename, delimiter)
    sim.draw_network()
    sim.simulation(p_trans, t_rec, t_sus, t_sim)
    sim.draw_network()

'''Esempio Simulazione Singola'''
#input parametri simulazione:p_trans, t_rec, t_sus, t_sim, num_sim 
random_simulation(25, 'data/facebook_combined.csv', ' ', 0.02, 7, 21, 60, 1)

'''Esempio Simulazione Random'''
#input parametri simulazione:p_trans, t_rec, t_sus, t_sim, num_sim
# random_simulation(25, 'facebook_combined.csv', ' ', 0.05, 7, 21, 60, 300)