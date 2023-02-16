from enum import Enum
from random import *
from utils import *
from statistics import mean, variance, mode
from copy import deepcopy
from plotting import *
from nlogo_colors import *
import itertools
import pandas as pd
import os
import numpy as np
from scipy.stats import chi2_contingency, truncnorm
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
from nlogo_io import *
# import statsmodels.formula.api as smf

DATA_DIR = 'D:/school/grad-school/Tufts/research/cog-cascades-trust'

"""
BELIEF ATTRIBUTES
"""

NUM_BELIEF_BUCKETS = 32

discrete = range(NUM_BELIEF_BUCKETS)

# class TestDiscrete7(Enum):
#   STRONG_DISBELIEF=12
#   DISBELIEF=1
#   MOD_DISBELIEF=2
#   UNCERTAIN=3
#   MOD_BELIEF=4
#   BELIEF=5
#   STRONG_BELIEF=6

class Attributes(Enum):
  A = discrete

def attrs_as_array(attr):
  # return map(lambda a: a.value, list(attr.value))
  return attr.value

"""
STATS STUFF
"""

'''
Return a tuple of summary statistical measures given a list of values.

:param l: The list to calculate stats for.
'''
def summary_statistics(l):
  if len(l) >= 2:
    return (mean(l), variance(l), mode(l))
  elif len(l) >= 1:
    return (mean(l), -1, mode(l))
  else:
    return (-1, -1, -1)

"""
Sample a distribution given a specific attribute. This distribution may
also depend on another, and if so, the function recursively calls
itself to return the needed dependency.

:param attr: An attribute from the Attribues enumeration to sample
its approriate distribution in the empirical data.
"""
def random_dist_sample(attr, resolution, given=None):
    return AttributeValues[attr.name]['vals'](resolution)

"""
Sample an attribute with an equal distribution over the values.

:param attr: The attribute to sample - e.g. Attributes.I.
"""
def random_sample(attr):
  rand = int(math.floor(random() * len(list(attr.value))))
  val = list(attr.value)[rand]
  return val.value

def test_random_sample():
  print(random_sample(Attributes.VG))


AttributeValues = {
  Attributes.A.name: {
    "vals": normal_dist,
    "depends_on": None
  }
}

# AttributeDistributions = {
#   Attributes.A.name: {
#     "dist": ADist,
#     "depends_on": None
#   }
# }

AttributeMAGThetas = {
  Attributes.A.name: {
    'default': AMAGDefaultTheta,
    'homophilic': AMAGHomophilicTheta,
    'heterophilic': AMAGHeterophilicTheta
  }   
}

"""
ANALYSIS FUNCTIONS
"""

def process_multiple_sim_data(path):
  for file in os.listdir(path):
    data = process_sim_data(f'{path}/{file}')
    stats = citizen_message_statistics(data[0], data[1])

'''
Parse a NetLogo chart export .csv file. This requires a single chart export file
and should not be run on an entire world export. This will return a dictionary
of plot points in a data frame, keyed by the pen name.

:param path: The path to the chart .csv file.
'''
def process_chart_data(path):
  f = open(path)
  raw = f.read()
  f.close()
  chunks = raw.split('\n\n')

  model_lines = chunks[1].replace('"','').split('\n')
  model_keys = model_lines[1].split(',')
  model_vals = model_lines[2].split(',')
  model_props = { model_keys[i]: model_vals[i] for i in range(len(model_keys)) }

  prop_lines = chunks[2].replace('"','').split('\n')
  chart_props = {}
  chart_props['color'] = {}
  keys = prop_lines[1].split(',')
  vals = prop_lines[2].split(',')
  for i in range(0, len(keys)):
    chart_props[keys[i]] = vals[i]

  data_sets = {}
  chart_lines = chunks[4].split('\n')
  
  data_set_splits = []
  split_line = chart_lines[0].split(',')
  for i in range(0, len(split_line)):
    el = split_line[i].replace('"','')
    if el != '':
      data_set_splits.append((el, i))
  for split in data_set_splits:
    data_sets[split[0]] = []

  for i in range(1, len(chart_lines)):
    line = chart_lines[i].replace('"','')
    if line == '': continue

    els = line.split(',')
    for j in range(0, len(data_set_splits)):
      split = data_set_splits[j]
      if j+1 == len(data_set_splits):
        data_sets[split[0]].append(els[split[1]:])
      else:
        data_sets[split[0]].append(els[split[1]:data_set_splits[j+1][1]])

  dfs = {}
  for split in data_set_splits:
    # df = pd.DataFrame(data=data_sets[split[0]][1:], columns=data_sets[split[0]][0])
    df = pd.DataFrame(data=data_sets[split[0]][1:], columns=data_sets[split[0]][0])
    del df['pen down?']
    chart_props['color'][split[0]] = df['color'].iloc[0] if len(df['color']) > 0 else 0
    del df['color']
    # del df['x']
    dfs[split[0]] = df

  return (model_props, chart_props, dfs)

'''
Read multiple NetLogo chart export files and plot all of them on a single
Matplotlib plot.

:param in_path: The directory to search for files in.
:param in_filename: A piece of filename that indicates which files to parse
in the process. This should usually be the name of the chart in the NetLogo file.
'''
def process_multi_chart_data(in_path, in_filename='percent-agent-beliefs'):
  props = []
  multi_data = []
  print(f'process_multi_chart_data for {in_path}/{in_filename}')
  if os.path.isdir(in_path):
    for file in os.listdir(in_path):
      if in_filename in file:
        data = process_chart_data(f'{in_path}/{file}')
        model_params = data[0]
        props.append(data[1])
        multi_data.append(data[2])

    full_data_size = int(model_params['tick-end']) + 1
    means = { key: [] for key in multi_data[0].keys() }
    for data in multi_data:
      for key in data.keys():
        data_vector = np.array(data[key]['y']).astype('float32')

        if len(data_vector) != full_data_size:
          # TODO: Need to do something here that reports the error
          print('ERROR parsing multi chart data -- data length did not equal number of ticks')
          continue

        if means[key] == []:
          means[key] = data_vector
        else:
          means[key] = np.vstack([means[key], data_vector])

    final_props = props[0]
    props_y_max = np.array([ float(prop['y max']) for prop in props ])
    final_props['y max'] = props_y_max.max()
    return (means, final_props, model_params)
  else:
    print(f'ERROR: Path not found {in_path}')
    return (-1, -1, -1)

'''
Given some multi-chart data, plot it and save the plot.

:param multi_data: Data with means and std deviations for each point.
:param props: Properties object for the plotting.
:param out_path: A path to save the results in.
:param out_filename: A filename to save results as, defaults to 'aggregate-chart'
:param show_plot: Whether or not to display the plot before saving.
'''
def plot_multi_chart_data(types, multi_data, props, out_path, out_filename='aggregate-chart', show_plot=False):
  if PLOT_TYPES.LINE in types:
    plot = plot_nlogo_multi_chart_line(props, multi_data)
    plt.savefig(f'{out_path}/{out_filename}_line.png')
    if show_plot: plt.show()
    plt.close()

  if PLOT_TYPES.STACK in types:
    plot = plot_nlogo_multi_chart_stacked(props, multi_data)
    plt.savefig(f'{out_path}/{out_filename}_stacked.png')
    if show_plot: plt.show()
    plt.close()

  if PLOT_TYPES.HISTOGRAM in types:
    plot = plot_nlogo_multi_chart_histogram(props, multi_data)
    plt.savefig(f'{out_path}/{out_filename}_histogram.png')
    if show_plot: plt.show()
    plt.close()

'''
Plot multiple NetLogo chart data sets on a single plot. 

:param props: The properties dictionary read in from reading the chart file. This
describes pen colors, x and y min and max, etc.
:param multi_data: A dictionary (keyed by line) of matrices where each row is one simulation's worth of data points.
'''
def plot_nlogo_multi_chart_stacked(props, multi_data):
  init_dist_width = 10

  # series = pd.Series(data)
  fig, (ax) = plt.subplots(1, figsize=(8,6))
  # ax, ax2 = fig.add_subplot(2)
  ax.set_ylim([0, 1])
  y_min = int(round(float(props['y min'])))
  y_max = int(round(float(props['y max'])))
  x_min = int(round(float(props['x min'])))
  x_max = int(round(float(props['x max'])))
  plt.yticks(np.arange(y_min, y_max+0.2, step=0.2))
  plt.xticks(np.arange(x_min, x_max+10, step=10))
  ax.set_ylabel("Portion of agents who believe b")
  ax.set_xlabel("Time Step")


  multi_data_keys_int = list(map(lambda el: int(el), multi_data.keys()))

  # To use Netlogo colors
  # line_color = lambda key: f"#{rgb_to_hex(NLOGO_COLORS[int(round(float(props['color'][key])))])}"

  # To use higher resolution colors
  resolution = int(max(multi_data_keys_int))+1
  line_color = lambda key: f"#{rgb_to_hex([ 255 - round((255/(resolution-1))*int(key)), 0, round((255/(resolution-1)) * int(key)) ])}"
  
  mean_vecs = []
  var_vecs = []
  rev_keys_int = sorted(multi_data_keys_int, reverse=True)
  rev_keys = list(map(lambda el: f'{el}', rev_keys_int))
  multi_data_has_multiple = lambda multi_data_entry: type(multi_data_entry[0]) == type(np.array(0)) and len(multi_data_entry) > 1
  for key in rev_keys:
    mean_vec = multi_data[key].mean(0) if multi_data_has_multiple(multi_data[key])  else multi_data[key]
    var_vec = multi_data[key].var(0) if multi_data_has_multiple(multi_data[key]) else np.zeros(len(mean_vec))

    # Add padding for the initial values so those are displayed in the graph
    mean_vec = np.insert(mean_vec, 0, [ mean_vec[0] for i in range(init_dist_width) ])

    mean_vecs.append(mean_vec)
    var_vecs.append(var_vec)
  
  ax.set_xlim([x_min-init_dist_width,len(mean_vecs[0])-init_dist_width])
  plt.stackplot(range(x_min-init_dist_width, len(mean_vecs[0])-init_dist_width), mean_vecs, colors=[ f'{line_color(c)}' for c in rev_keys ], labels=[ f'b = {b}' for b in rev_keys ])

'''
Plot multiple NetLogo chart data sets on a single plot. This will scatterplot
each data set and then draw a line of the means at each point through the
entire figure.

:param props: The properties dictionary read in from reading the chart file. This
describes pen colors, x and y min and max, etc.
:param multi_data: A list of dataframes that contain chart data.
'''
def plot_nlogo_multi_chart_line(props, multi_data):
  # series = pd.Series(data)
  # print(multi_data)
  fig, (ax) = plt.subplots(1, figsize=(8,6))
  # ax, ax2 = fig.add_subplot(2)
  y_min = int(round(float(props['y min'])))
  y_max = int(round(float(props['y max'])))
  x_min = int(round(float(props['x min'])))
  x_max = int(round(float(props['x max'])))
  ax.set_ylim([0, y_max])
  plt.yticks(np.arange(y_min, y_max, step=1))
  # plt.yticks(np.arange(y_min, y_max*1.1, step=y_max/10))
  plt.xticks(np.arange(x_min, x_max*1.1, step=5))
  ax.set_ylabel("% of agents who believe b")
  ax.set_xlabel("Time Step")

  line_color = lambda key: '#000000'

  if list(multi_data.keys())[0] != 'default':
    # This is specific code to set the colors for belief resolutions
    multi_data_keys_int = list(map(lambda el: int(el), multi_data.keys()))
    resolution = int(max(multi_data_keys_int))+1
    line_color = lambda key: f"#{rgb_to_hex([ 255 - round((255/max(resolution-1,1))*int(key)), 0, round((255/max(resolution-1,1)) * int(key)) ])}"
 
  multi_data_has_multiple = lambda multi_data_entry: type(multi_data_entry[0]) == type(np.array(0)) and len(multi_data_entry) > 1
 
  for key in multi_data:
    mean_vec = multi_data[key].mean(0) if multi_data_has_multiple(multi_data[key])  else multi_data[key]
    var_vec = multi_data[key].var(0) if multi_data_has_multiple(multi_data[key]) else np.zeros(len(mean_vec))
    # print(multi_data[key])
    # print(var_vec)
    ax.plot(mean_vec, c=line_color(key))
    ax.fill_between(range(x_min, len(mean_vec)), mean_vec-var_vec, mean_vec+var_vec, facecolor=f'{line_color(key)}44')
  
  return multi_data

'''
Plot multiple NetLogo chart data sets on a single plot. This will scatterplot
each data set and then draw a line of the means at each point through the
entire figure.

:param props: The properties dictionary read in from reading the chart file. This
describes pen colors, x and y min and max, etc.
:param multi_data: A list of dataframes that contain chart data.
'''
def plot_nlogo_histogram(props, multi_data):
  # series = pd.Series(data)
  # print(multi_data)
  fig, (ax) = plt.subplots(1, figsize=(8,6))
  # ax, ax2 = fig.add_subplot(2)
  ax.set_ylim([0, 1.1])
  y_min = int(round(float(props['y min'])))
  y_max = int(round(float(props['y max'])))
  x_min = int(round(float(props['x min'])))
  x_max = int(round(float(props['x max'])))
  plt.yticks(np.arange(y_min, y_max+0.2, step=0.2))
  plt.xticks(np.arange(x_min, x_max+10, step=10))
  ax.set_ylabel("# of agents who believe b")
  ax.set_xlabel("Time Step")

  line_color = lambda key: '#000000'

  if list(multi_data.keys())[0] != 'default':
    # This is specific code to set the colors for belief resolutions
    multi_data_keys_int = list(map(lambda el: int(el), multi_data.keys()))
    resolution = int(max(multi_data_keys_int))+1
    bar_color = lambda key: f"#{rgb_to_hex([ 255 - round((255/max(resolution-1,1))*int(key)), 0, round((255/max(resolution-1,1)) * int(key)) ])}"

  multi_data_has_multiple = lambda multi_data_entry: type(multi_data_entry[0]) == type(np.array(0)) and len(multi_data_entry) > 1
 
  for key in multi_data:
    mean_vec = multi_data[key].mean(0) if multi_data_has_multiple(multi_data[key])  else multi_data[key]
    var_vec = multi_data[key].var(0) if multi_data_has_multiple(multi_data[key]) else np.zeros(len(mean_vec))
    # print(var_vec)
    ax.plot(mean_vec, c=bar_color(key))
    ax.fill_between(range(x_min, len(mean_vec)), mean_vec-var_vec, mean_vec+var_vec, facecolor=f'{bar_color(key)}44')
  
  return multi_data
      
'''
From a NetLogo world export file, read in the simulation data for citizens
and media entities. They are stored in Pandas dataframes for further processing.

:param path: The path to the file to read data in from.
'''
def process_sim_data(path):
  f = open(path)
  raw = f.read()
  f.close()
  lines = raw.split('\n')
  turtle_data = []
  for i in range(0, len(lines)):
    line = lines[i]
    if line == '"TURTLES"':
      while line.strip() != '':
        i += 1
        line = lines[i]
        turtle_data.append(line.replace('""','"').split(','))

  turtle_data[0] = list(map(lambda el: el.replace('"',''), turtle_data[0]))
  turtle_df = pd.DataFrame(data=turtle_data[1:], columns=turtle_data[0])

  unneeded_cols = ['color', 'heading', 'xcor', 'ycor', 'label', 'label-color', 'shape', 'pen-size', 'pen-mode', 'size','hidden?']
  citizen_delete = ['media-attrs','messages-sent']
  media_delete = ['messages-heard','brain','messages-believed']

  for col in unneeded_cols:
    del turtle_df[col]

  citizen_df = turtle_df[turtle_df['breed'] == '"{breed citizens}"']
  media_df = turtle_df[turtle_df['breed'] == '"{breed medias}"']

  for col in citizen_delete:
    del citizen_df[col]
  for col in media_delete:
    del media_df[col]

  return (citizen_df, media_df)

'''
Get a relevant set of statistics about the citizens' ending state
after the simulation was run: which messages they heard

:param citizen_df: A dataframe containing citizen data.
'''
def citizen_message_statistics(citizen_df, media_df):
  messages = {}
  # Generate a data frame for media messages
  for m in media_df.iterrows():
    m_sent = nlogo_mixed_list_to_dict(m[1]['messages-sent'])
    messages.update(m_sent)
  for m_id, val in messages.items():
    val['id'] = int(m_id.strip())

  message_vals = list(messages.values())
  messages_df = pd.DataFrame(data=message_vals, columns=list(message_vals[0].keys()))

  # Generate citizen data frames relevant for statistics
  heard_dfs = {}
  for citizen in citizen_df.iterrows():
    parsed = nlogo_mixed_list_to_dict(citizen[1]['messages-heard'])
    flat_heard = []
    for timestep,message_ids in parsed.items():
      flat_heard.extend([ { 'tick': int(timestep), 'message_id': m_id  } for m_id in message_ids ] )
    df = pd.DataFrame(flat_heard)
    heard_dfs[int(citizen[1]['who'].replace('"',''))] = df
  
  believed_dfs = {}
  for citizen in citizen_df.iterrows():
    parsed = nlogo_mixed_list_to_dict(citizen[1]['messages-believed'])
    flat_believed = []
    if not type(parsed) == list:
      for timestep,message_ids in parsed.items():
        flat_believed.extend([ { 'tick': int(timestep), 'message_id': m_id  } for m_id in message_ids ] )
      df = pd.DataFrame(flat_believed)
      believed_dfs[int(citizen[1]['who'].replace('"',''))] = df
    else:
      believed_dfs[int(citizen[1]['who'].replace('"',''))] = pd.DataFrame()
  
  # Analyze the data frames for some statistical measures (per citizen)
  # - Total heard
  # - Total believed
  # - Ratio of believed/heard
  # - Totals heard broken down by partisanship & ideology
  # - Totals believed broken down by partisanship & ideology
  # - Totals heard broken down by virus belief
  # - Totals believed broken down by virus belief
  # - Somehow get at beliefs over time?

  per_cit_stats = {}
  for row in citizen_df.iterrows():
    citizen = row[1]
    cit_id = int(citizen['who'].replace('"',''))
    per_cit_stats[cit_id] = per_citizen_stats(cit_id, messages_df, heard_dfs[cit_id], believed_dfs[cit_id])

  # Analyze some group-level measures
  # - Aggregate by citizen's partisan/ideology pair
  # - 

  aggregate_stats = citizens_stats(citizen_df, per_cit_stats)

  return (messages_df, heard_dfs, believed_dfs, per_cit_stats, aggregate_stats)

'''
Generate some statistical measures based on the aggregate view of the citizenry.

:param cit_df: A dataframe containing data for each citizen.
:param per_cit_stats: A dictionary of statistical measures calculated
for each citizen. This is generated from the `citizen_stats()` function.
'''
def citizens_stats(cit_df, per_cit_stats):
  partisanships = list(attrs_as_array(Attributes.P))
  ideologies = list(attrs_as_array(Attributes.I))
  virus_believe_vals = [ -1, 0, 1 ]

  pi_keyed_dict = { (prod[0], prod[1]): 0 for prod in itertools.product(partisanships, ideologies) }
  virus_belief_keyed_dict = { (prod[0], prod[1], prod[2]): 0 for prod in itertools.product(virus_believe_vals, repeat=3) }

  total_by_p = { p: 0 for p in partisanships }
  total_by_i = { i: 0 for i in ideologies }
  total_by_p_i = pi_keyed_dict.copy()

  heard_by_cit_p_i = { (prod[0], prod[1]): [] for prod in itertools.product(partisanships, ideologies) }
  believed_by_cit_p_i = deepcopy(heard_by_cit_p_i)
  # This will be a dict of { (citizen_p, citizen_i): { (message_p, message_i):
  # [ list of (p,i) message heard ] } } - i.e. a dictionary of message types
  # heard by political group
  heard_by_pi_given_cit_pi = { (prod[0], prod[1]): deepcopy(heard_by_cit_p_i) for prod in itertools.product(partisanships, ideologies) }
  believed_by_pi_given_cit_pi = { (prod[0], prod[1]): deepcopy(heard_by_cit_p_i) for prod in itertools.product(partisanships, ideologies) }

  virus_bel_counts = virus_belief_keyed_dict.copy()
  virus_bel_totals = { (prod[0], prod[1], prod[2]): [] for prod in itertools.product(virus_believe_vals, repeat=3) }

  ending_beliefs_by_p_i = { (prod[0], prod[1]): virus_belief_keyed_dict.copy() for prod in itertools.product(partisanships, ideologies) }
  # Similarly to above, this will be a dict of { (cit_p, cit_i): {
  # (virus_beliefs...): [ list of per-citizen (vg,vs,vd) messages heard ] } }
  heard_by_virus_bel_given_pi = { (prod[0], prod[1]): deepcopy(virus_bel_totals) for prod in itertools.product(partisanships, ideologies) }
  believed_by_virus_bel_given_pi = { (prod[0], prod[1]): deepcopy(virus_bel_totals) for prod in itertools.product(partisanships, ideologies) }

  for cit in cit_df.iterrows():
    citizen = cit[1]
    cit_id = int(citizen['who'].replace('"',''))
    brain = nlogo_mixed_list_to_dict(citizen['brain'])
    stats = per_cit_stats[cit_id]

    pi_tup = pi_tuple(brain)
    virus_tup = virus_tuple(brain)

    total_by_i[int(brain['I'])] += 1
    total_by_p[int(brain['P'])] += 1
    total_by_p_i[pi_tup] += 1
    heard_by_cit_p_i[pi_tup].append(stats['total_heard'])
    believed_by_cit_p_i[pi_tup].append(stats['total_believed'])
    for message_pi_tup in stats['heard_by_p_i'].keys():
      heard_by_pi_given_cit_pi[pi_tup][message_pi_tup].append(stats['heard_by_p_i'][message_pi_tup])
      believed_by_pi_given_cit_pi[pi_tup][message_pi_tup].append(stats['believed_by_p_i'][message_pi_tup])

    virus_bel_counts[virus_tup] += 1
    ending_beliefs_by_p_i[pi_tup][virus_tup] += 1
    for message_virus_tup in stats['heard_by_virus_bel'].keys():
      heard_by_virus_bel_given_pi[pi_tup][message_virus_tup].append(stats['heard_by_virus_bel'][message_virus_tup])
      believed_by_virus_bel_given_pi[pi_tup][message_virus_tup].append(stats['believed_by_virus_bel'][message_virus_tup])
  
  heard_sum_by_p_i = { pi: summary_statistics(heard_by_cit_p_i[pi]) for pi in heard_by_cit_p_i.keys() }
  believed_sum_by_p_i = { pi: summary_statistics(believed_by_cit_p_i[pi]) for pi in believed_by_cit_p_i.keys() }

  heard_sum_by_pi_given_pi = pi_keyed_dict.copy()
  for pi in heard_by_pi_given_cit_pi.keys():
    entry = heard_by_pi_given_cit_pi[pi]
    heard_sum_by_pi_given_pi[pi] = { cit_pi: summary_statistics(entry[cit_pi]) for cit_pi in entry.keys() }

  believed_sum_by_pi_given_pi = pi_keyed_dict.copy()
  for pi in believed_by_pi_given_cit_pi.keys():
    entry = believed_by_pi_given_cit_pi[pi]
    believed_sum_by_pi_given_pi[pi] = { cit_pi: summary_statistics(entry[cit_pi]) for cit_pi in entry.keys() }

  heard_sum_by_virus_given_pi = pi_keyed_dict.copy()
  for pi in heard_by_virus_bel_given_pi.keys():
    entry = heard_by_virus_bel_given_pi[pi]
    heard_sum_by_virus_given_pi[pi] = { virus_bel: summary_statistics(entry[virus_bel]) for virus_bel in entry.keys() }

  believed_sum_by_virus_given_pi = pi_keyed_dict.copy()
  for pi in believed_by_virus_bel_given_pi.keys():
    entry = believed_by_virus_bel_given_pi[pi]
    believed_sum_by_virus_given_pi[pi] = { virus_bel: summary_statistics(entry[virus_bel]) for virus_bel in entry.keys() }
  
  stats_given_pi = pi_keyed_dict.copy()
  for pi in stats_given_pi.keys():
    stats_given_pi[pi] = {}
    stats_given_pi[pi]['n'] = total_by_p_i[pi]
    stats_given_pi[pi]['total_heard'] = heard_sum_by_p_i[pi]
    stats_given_pi[pi]['total_believed'] = believed_sum_by_p_i[pi]
    stats_given_pi[pi]['ending_beliefs'] = ending_beliefs_by_p_i[pi]
    stats_given_pi[pi]['heard_stats_by_pi'] = heard_sum_by_pi_given_pi[pi]
    stats_given_pi[pi]['believed_stats_by_pi'] = believed_sum_by_pi_given_pi[pi]
    stats_given_pi[pi]['heard_stats_by_virus'] = heard_sum_by_virus_given_pi[pi]
    stats_given_pi[pi]['believed_stats_by_virus'] = believed_sum_by_virus_given_pi[pi]

  return stats_given_pi

'''
Generate some statistics for each citizen in the simulation report data.
Measures that are reported:
- Total messages heard and believed
- Believed/heard ratio
- Messages heard & believed by (partisan,ideology) pair
- Messages heard & believed by virus-belief combination
'''
def per_citizen_stats(cit_id, messages_df, heard_df, believed_df):
  cit_stats = {}
  cit_stats['total_heard'] = len(heard_df)
  cit_stats['total_believed'] = len(believed_df)
  cit_stats['bel_heard_ratio'] = cit_stats['total_believed']/cit_stats['total_heard']

  partisanships = list(attrs_as_array(Attributes.P))
  ideologies = list(attrs_as_array(Attributes.I))
  heard_by_p_i = { (prod[0], prod[1]): 0 for prod in itertools.product(partisanships, ideologies) }

  for row in heard_df.iterrows():
    heard = row[1]
    m_id = int(heard['message_id'])
    message = messages_df[messages_df['id'] == m_id]
    heard_by_p_i[pi_tuple(message)] += 1

  # (P, I) tuples
  believed_by_p_i = { (prod[0], prod[1]): 0 for prod in itertools.product(partisanships, ideologies) }
  for row in believed_df.iterrows():
    believed = row[1]
    m_id = int(believed['message_id'])
    message = messages_df[messages_df['id'] == m_id]
    believed_by_p_i[pi_tuple(message)] += 1
  cit_stats['heard_by_p_i'] = heard_by_p_i
  cit_stats['believed_by_p_i'] = believed_by_p_i

  # (VG, VG, VD) tuples
  virus_believe_vals = [ -1, 0, 1 ]
  heard_by_virus_bel = { (prod[0], prod[1], prod[2]): 0 for prod in itertools.product(virus_believe_vals, repeat=3) }
  for row in heard_df.iterrows():
    heard = row[1]
    m_id = int(heard['message_id'])
    message = messages_df[messages_df['id'] == m_id]
    heard_by_virus_bel[virus_tuple(message)] += 1
  believed_by_virus_bel = { (prod[0], prod[1], prod[2]): 0 for prod in itertools.product(virus_believe_vals, repeat=3) }
  for row in believed_df.iterrows():
    believed = row[1]
    m_id = int(believed['message_id'])
    message = messages_df[messages_df['id'] == m_id]
    believed_by_virus_bel[virus_tuple(message)] += 1
  cit_stats['heard_by_virus_bel'] = heard_by_virus_bel
  cit_stats['believed_by_virus_bel'] = believed_by_virus_bel

  return cit_stats

def group_stats_by_attr(group_stats, attr):
  return { pi: val[attr] for pi,val in group_stats.items() }

'''
Return a tuple of partisanship and ideology attributes from a given object.

:param obj: Some object to fetch parameters from.
'''
def pi_tuple(obj): return (int(obj['P']),int(obj['I']))

'''
Return a tuple of virus-related beliefs from a given object.

:param obj: Some object to fetch parameters from.
'''
def virus_tuple(obj): return (int(obj['VG']),int(obj['VS']),int(obj['VD']))

def plot_stats_means(stats_data, title, path):
  plot_and_save_series({ key: val[0] for key,val in stats_data.items() }, title, path, 'bar')

def pi_data_charts(stats_data, attr, replace, title_w_replace, path_w_replace):
  partisanships = list(attrs_as_array(Attributes.P))
  ideologies = list(attrs_as_array(Attributes.I))
  pi_keys = { (prod[0], prod[1]): 0 for prod in itertools.product(partisanships, ideologies) }
  for key in pi_keys:
    plot_stats_means(stats_data[key][attr], title_w_replace.replace(replace, str(key)), path_w_replace.replace(replace, f'{key[0]}-{key[1]}'))

def corr_multi_data(multi_data_1, multi_data_2, method='pearson'):
  '''
  Calculate correlations between two sets of multi data.

  :param multi_data_1: A first set of data over multiple simulation runs, keyed by agent belief value.
  :param multi_data_2: A second set of data over multiple simulation runs, keyed by agent belief value.

  :return: Correlation values per belief value.
  '''
  m1_means = { key: multi_data_1[key].mean(0) for key in multi_data_1 }
  m2_means = { key: multi_data_2[key].mean(0) for key in multi_data_2 }

  rs = {}
  for key in multi_data_1:
    df = pd.DataFrame({ 'data1': m1_means[key], 'data2': m2_means[key] })
    # Uncomment if you need to investigate the df
    # rs[key] = {}
    # rs[key]['df'] = df
    # rs[key]['corr'] = df.corr(method=method).iloc[0,1]
    rs[key] = df.corr(method=method).iloc[0,1]
  return rs

def aggregate_corr(corr_by_bel):
  '''
  Generate an average correlation across correlations by belief value.

  :param corr_by_bel: A dictionary keyed by belief value of correlation values.
  '''
  non_nan = np.array(list(corr_by_bel.values()))
  non_nan = non_nan[np.logical_not(np.isnan(non_nan))]
  return non_nan.sum() / len(non_nan)

def chi_sq_test_multi_data(multi_data_1, multi_data_2, N):
  '''
  Perform a chi squared test on two sets of multi data for each timestep in the simulation data. 

  NOTE: This converts agent population percentages to total numbers and pads by 1
  in order to circumnavigate sampling 0 agents.

  :param multi_data_1: A first set of data over multiple simulation runs, keyed by agent belief value.
  :param multi_data_2: A second set of data over multiple simulation runs, keyed by agent belief value.
  :param N: The number of agents in the simulation.

  :returns: Returns the chi2 timeseries data.
  '''

  m1_means = [ multi_data_1[key].mean(0) for key in multi_data_1 ]
  m2_means = [ multi_data_2[key].mean(0) for key in multi_data_2 ]

  data = []
  for timestep in range(len(m1_means[0])):
    data.append([])
    # Append on lists of the values for each belief at timestep t 
    data[timestep].append([ m1_means[bel][timestep] for bel in range(len(m1_means)) ])
    data[timestep].append([ m2_means[bel][timestep] for bel in range(len(m2_means)) ])
  
  for data_t in data:
    for i in range(len(data_t[0])):
      data_t[0][i] = round(N * data_t[0][i] + 1)
      data_t[1][i] = round(N * data_t[1][i] + 1)
  
  chi2_data = [ chi2_contingency(data_t) for data_t in data ]
  # TODO: CHANGE THIS BACK
  # return chi2_data
  return (data, chi2_data)

def chi_sq_global(chi2_data):
  '''
  Convert a timeseries of chi squared test data into a global measure of how many
  entries in the time series are statistically independent. Higher values indicate
  higher levels of independence.

  :param chi2_data: An array of timeseries chi squared data from the scipy test.
  '''
  data = np.array([ el[1] for el in chi2_data ])
  return (data <= 0.05).sum() / len(data)


def plot_chi_sq_data(chi2_data, props, title, out_path, out_filename):
  '''
  Plot a time series calculation of chi squared measures per timestep.

  :param chi2_data: The timeseries data from running chi squared stats on belief data.
  :param props: The simulation properties to pull time data from.
  :param title: Text to title the plot with.
  '''
  # series = pd.Series(data)
  fig, (ax) = plt.subplots(1, figsize=(8,6))
  # ax, ax2 = fig.add_subplot(2)
  ax.set_ylim([0, 1.0])
  y_min = 0
  y_max = 1.0
  x_min = int(props['x min'])
  x_max = int(props['x max'])
  plt.yticks(np.arange(y_min, y_max+0.2, step=0.05))
  plt.xticks(np.arange(x_min, x_max+10, step=10))
  ax.set_ylabel("p value")
  ax.set_xlabel("Time Step")
  ax.set_title(f'{title}')
 
  ax.plot([ data[1] for data in chi2_data ])
  plt.savefig(f'{out_path}/{out_filename}')
  plt.close()

"""
##################
EXPERIMENT-SPECIFIC
ANALYSIS
##################
"""

class PLOT_TYPES(Enum):
  LINE = 0
  STACK = 1
  HISTOGRAM = 2

def process_exp_outputs(param_combos, plots, path):
  '''
  Process the output of a NetLogo experiment, aggregating all results
  over simulation runs and generating plots for them according to
  all the parameter combinations denoted in param_combos.
  
  :param param_combos: A list of parameters where their values are
  lists (e.g. [ ['simple','complex'], ['default', 'gradual'] ])
  :param plots: A list of dictionaries keyed by the name of the NetLogo
  plot to process, with value of a list of PLOT_TYPE
  (e.g. { 'polarization': [PLOT_TYPES.LINE], 'agent-beliefs': [...] })
  :param path: The root path to begin processing in.
  '''
  combos = []
  for combo in itertools.product(*param_combos):
    combos.append(combo)

  if not os.path.isdir(f'{path}/results'):
    os.mkdir(f'{path}/results')

  for combo in combos:
    for (plot_name, plot_types) in plots.items():
      # print(plot_name, plot_types)
      (multi_data, props, model_params) = process_multi_chart_data(f'{path}/{"/".join(combo)}', plot_name)
      # If there was no error processing the data
      if multi_data != -1:
        plot_multi_chart_data(plot_types, multi_data, props, f'{path}/results', f'{"-".join(combo)}_{plot_name}-agg-chart')

def get_all_multidata(param_combos, plots, path):
  combos = []
  for combo in itertools.product(*param_combos):
    combos.append(combo)

  if not os.path.isdir(f'{path}/results'):
    os.mkdir(f'{path}/results')

  multi_datas = {}
  for combo in combos:
    for (plot_name, plot_types) in plots.items():
      # print(plot_name, plot_types)
      (multi_data, props, model_params) = process_multi_chart_data(f'{path}/{"/".join(combo)}', plot_name)
      multi_datas[(combo,plot_name)] = multi_data
  return multi_datas

def process_parameter_sweep_test_exp(path):
  cognitive_translate = ['0', '1', '2']
  institution_tactic = ['broadcast-brain', 'appeal-mean']
  media_ecosystem_n = ['20']
  media_ecosystem_dist = [ 'uniform', 'normal', 'polarized' ]
  init_cit_dist = ['normal', 'uniform', 'polarized']
  zeta_media = ['0.25','0.5','0.75','1']
  zeta_cit = ['0.25','0.5','0.75','1']
  citizen_memory_length = ['5']
  ba_m = ['3']
  graph_type = ['barabasi-albert']
  repetition = list(map(str, range(2)))

  process_exp_outputs(
    [cognitive_translate,institution_tactic,media_ecosystem_dist,media_ecosystem_n,init_cit_dist,zeta_media,zeta_cit,citizen_memory_length,graph_type,ba_m,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'polarization': [PLOT_TYPES.LINE]},
    # 'polarization': [PLOT_TYPES.LINE],
    # 'homophily': [PLOT_TYPES.LINE]},
    path)

def process_parameter_sweep_tinytest_exp(path):
  cognitive_translate = ['0', '1', '2']
  epsilon = ['0']
  institution_tactic = ['broadcast-brain', 'appeal-mean']
  media_ecosystem_n = ['20']
  media_ecosystem_dist = [ 'uniform', 'normal', 'polarized' ]
  init_cit_dist = ['normal', 'uniform', 'polarized']
  citizen_memory_length = ['5']
  repetition = list(map(str, range(2)))

  process_exp_outputs(
    [cognitive_translate,institution_tactic,media_ecosystem_dist,media_ecosystem_n,init_cit_dist,epsilon,citizen_memory_length,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'polarization': [PLOT_TYPES.LINE]},
    # 'polarization': [PLOT_TYPES.LINE],
    # 'fragmentation': [PLOT_TYPES.LINE],
    # 'homophily': [PLOT_TYPES.LINE]},
    path)



def get_conditions_to_polarization_multidata(path):
  cognitive_translate = ['0', '1', '2']
  epsilon = ['0', '1', '2']
  institution_tactic = ['broadcast-brain', 'appeal-mean', 'appeal-median']
  media_ecosystem_dist = [ 'uniform', 'normal', 'polarized' ]
  # ba_m = ['3', '5', '10']
  ba_m = ['3' ]
  graph_types = [ 'ba-homophilic', 'barabasi-albert' ]
  init_cit_dist = ['normal', 'uniform', 'polarized']
  repetition = list(map(str, range(2)))

  return get_all_multidata(
    [cognitive_translate,institution_tactic,media_ecosystem_dist,init_cit_dist,epsilon,graph_types,ba_m,repetition],
    {'percent-agent-beliefs': [PLOT_TYPES.LINE, PLOT_TYPES.STACK],
    'polarization': [PLOT_TYPES.LINE],
    'disagreement': [PLOT_TYPES.LINE],
    'homophily': [PLOT_TYPES.LINE],
    'chi-sq-cit-media': [PLOT_TYPES.LINE]},
    path)

def logistic_regression_polarization(polarization_data):
  '''
  Run a logistic regression to fit polarization data given different
  combinations of simulation parameters.

  This analysis supports results in Rabb & Cowen, 2022 in Section 5
  where we discuss the effect of parameters on polarization results
  reported in Tables 3-5.

  :param polarization_data: The result of polarization_analysis(multidata)
  This contains 2 key dataframes -- one for polarizing results, one for
  nonpolarizing ones
  '''
  polarizing = polarization_data['polarizing']
  nonpolarizing = polarization_data['nonpolarizing']
  polarizing['polarized'] = 1
  nonpolarizing['polarized'] = 0
  
  df = polarizing.append(nonpolarizing)
  df['epsilon'] = df['epsilon'].astype("int64")
  df['translate'] = df['translate'].astype("int64")
  df['graph_type'] = df['graph_type'].astype("category")
  df['tactic'] = df['tactic'].astype("category")
  df['citizen_dist'] = df['citizen_dist'].astype("category")
  df['media_dist'] = df['media_dist'].astype("category")

  # This model yields results discussed in Subsection 5.1, the effect
  # of h_G, epsilon, and gamma on polarization results.
  result = smf.logit("polarized ~ epsilon + translate + graph_type", data=df).fit()
  print(result.summary())

  # This model yields results discussed in Subsection 5.2, Table 5,
  # the effect of C, I and gamma on polarization results. To select
  # I = 'uniform' or I='polarized', different lines can be commented
  # or uncommented.
  df = df[df['tactic']=='broadcast-brain']
  # df = df[df['media_dist']=='uniform']
  df = df[df['media_dist']=='polarized']
  result = smf.logit("polarized ~ epsilon + translate + graph_type + citizen_dist", data=df).fit()

  print(result.summary())

def polarization_results_by_tactic_exposure(polarization_data):
  '''
  Run an analysis to see how many results polarized vs nonpolarized for
  parameter combinations of varphi (tactic) and gamma (translate).

  This analysis supports Table 4 in Rabb & Cowen 2022.
  
  :param polarization_data: The result of polarization_analysis(multidata)
  This contains 2 key dataframes -- one for polarizing results, one for
  nonpolarizing ones
  '''
  polarizing = polarization_data['polarizing']
  nonpolarizing = polarization_data['nonpolarizing']
  polarizing['polarized'] = 1
  nonpolarizing['polarized'] = 0
  all_results = polarizing.append(nonpolarizing)

  dfs = {
    "all_broadcast": all_results[all_results['tactic'] == 'broadcast-brain'],
    "all_mean": all_results[all_results['tactic'] == 'appeal-mean'],
    "all_median": all_results[all_results['tactic'] == 'appeal-median'],
  }

  gamma_values = [0,1,2]
  proportions = {}
  for (df_name, df) in dfs.items():
    print(f'{df_name}\n==========')
    for gamma in gamma_values:
      partition_polarized = df.query(f'translate=="{gamma}" and polarized==1')
      partition_nonpolarized = df.query(f'translate=="{gamma}" and polarized==0')
      partition_all = df.query(f'translate=="{gamma}"')
      
      # Use this line to report percent of results that are polarized
      proportions[(gamma)] = {'polarized': len(partition_polarized) / len(partition_all), 'nonpolarized': len(partition_nonpolarized) / len(partition_all) }
      
      # Use this line to report number of results that are polarized
      # proportions[(gamma)] = {'polarized': len(partition_polarized), 'nonpolarized': len(partition_nonpolarized) }


def polarization_results_by_fragmentation_exposure(polarization_data):
  '''
  Run an analysis to see how many results polarized vs nonpolarized for
  parameter combinations of h_G (homophily), epislon, and gamma (translate).

  This analysis supports Table 3 in Rabb & Cowen 2022.
  
  :param polarization_data: The result of polarization_analysis(multidata)
  This contains 2 key dataframes -- one for polarizing results, one for
  nonpolarizing ones
  '''
  polarizing = polarization_data['polarizing']
  nonpolarizing = polarization_data['nonpolarizing']
  polarizing['polarized'] = 1
  nonpolarizing['polarized'] = 0
  all_results = polarizing.append(nonpolarizing)

  dfs = {
    "all_regular": all_results[all_results['graph_type'] == 'barabasi-albert'],
    "all_homophilic": all_results[all_results['graph_type'] == 'ba-homophilic'],
  }

  epsilon_values = [0,1,2]
  gamma_values = [0,1,2]
  proportions = {}
  for (df_name, df) in dfs.items():
    print(f'{df_name}\n==========')
    for epsilon in epsilon_values:
      for gamma in gamma_values:
        partition_polarized = df.query(f'epsilon=="{epsilon}" and translate=="{gamma}" and polarized==1')
        partition_nonpolarized = df.query(f'epsilon=="{epsilon}" and translate=="{gamma}" and polarized==0')
        partition_all = df.query(f'epsilon=="{epsilon}" and translate=="{gamma}"')

        # Use this line to report percent of results that are polarized
        proportions[(epsilon,gamma)] = {'polarized': len(partition_polarized) / len(partition_all), 'nonpolarized': len(partition_nonpolarized) / len(partition_all) }

        # Use this line to report number of results that are polarized
        # proportions[(epsilon,gamma)] = {'polarized': len(partition_polarized), 'nonpolarized': len(partition_nonpolarized) }

def polarization_results_by_broadcast_distributions(polarization_data):
  '''
  Run an analysis to see how many results polarized vs nonpolarized for
  parameter gamma (translate), citizen distribution and institution
  distribution.

  This analysis supports Table 5 in Rabb & Cowen 2022.
  
  :param polarization_data: The result of polarization_analysis(multidata)
  This contains 2 key dataframes -- one for polarizing results, one for
  nonpolarizing ones
  '''
  polarizing = polarization_data['polarizing']
  nonpolarizing = polarization_data['nonpolarizing']
  polarizing['polarized'] = 1
  nonpolarizing['polarized'] = 0
  all_results = polarizing.append(nonpolarizing)

  dfs = {
    "translate_0": all_results.query("translate=='0' and tactic=='broadcast-brain'"),
    "translate_1": all_results.query("translate=='1' and tactic=='broadcast-brain'"),
    "translate_2": all_results.query("translate=='2' and tactic=='broadcast-brain'"),
  }

  dist_values = ['uniform','normal','polarized']
  proportions = {}
  for (df_name, df) in dfs.items():
    print(f'{df_name}\n==========')
    for cit_dist in dist_values:
      for inst_dist in dist_values:
        partition_polarized = df.query(f'citizen_dist=="{cit_dist}" and media_dist=="{inst_dist}" and polarized==1')
        partition_nonpolarized = df.query(f'citizen_dist=="{cit_dist}" and media_dist=="{inst_dist}" and polarized==0')
        partition_all = df.query(f'citizen_dist=="{cit_dist}" and media_dist=="{inst_dist}"')

        # Use this line to report percent of results that are polarized
        proportions[(cit_dist,inst_dist)] = {'polarized': len(partition_polarized) / len(partition_all), 'nonpolarized': len(partition_nonpolarized) / len(partition_all) }

        # Use this line to report number of results that are polarized
        # proportions[(cit_dist,inst_dist)] = {'polarized': len(partition_polarized), 'nonpolarized': len(partition_nonpolarized) }

    print(proportions)
 
def polarization_stability_analysis(multidata):
  '''
  Analyze each individual run of the polarization experiment
  to see if its individual runs polarization result match
  that of the mean of the results.

  This analysis supports the second to last paragraph of Section 4
  in Rabb & Cowen, 2022.

  :param multidata: Multidata gathered from the experiment.
  '''
  threshold = 0.01
  stability_df = pd.DataFrame(columns=['translate','tactic','media_dist','citizen_dist','epsilon','graph_type','ba-m','repetition','polarized?','polarizing','nonpolarizing','ratio_match'])

  polarization_data = { key: value for (key,value) in multidata.items() if key[1] == 'polarization' }
  polarization_means = { key: value['0'].mean(0) for (key,value) in polarization_data.items() }
  x = np.array([[val] for val in range(len(list(polarization_means.values())[0]))])

  for (param_combo, data) in polarization_data.items():
    polarizing = []
    nonpolarizing = []
    for run_data in data['0']:
      model = LinearRegression().fit(x, run_data)
      if model.coef_[0] >= threshold:
        polarizing.append(run_data)
      elif model.coef_[0] <= threshold * -1:
        nonpolarizing.append(run_data)
      elif model.intercept_ >= 8.5:
        polarizing.append(run_data)
      else:
        nonpolarizing.append(run_data)
    polarizing_ratio = len(polarizing) / len(data['0'])
    nonpolarizing_ratio = len(nonpolarizing) / len(data['0'])

    model = LinearRegression().fit(x, polarization_means[param_combo])
    polarized = model.coef_[0] >= threshold or (model.coef_[0] < threshold and model.coef_[0] > (threshold * -1) and model.intercept_ >= 8.5)
    match = polarizing_ratio if polarized else nonpolarizing_ratio

    stability_df.loc[len(stability_df.index)] = list(param_combo[0]) + [polarized, polarizing_ratio, nonpolarizing_ratio,match]
  
  diffs = [0.2, 0.4, 0.6, 0.8, 1]
  diff_parts = { diff: stability_df[round(abs(stability_df['polarizing']-stability_df['nonpolarizing']),1) == diff] for diff in diffs }
  diff_parts_span = { diff: { col: df[col].unique() for col in df.columns } for (diff, df) in diff_parts.items() }
  
  return { 'stability': stability_df, 'diff_parts': diff_parts, 'diff_span': diff_parts_span }

def polarization_analysis(multidata):
  '''
  Analyze polarization data for any of the experiments' multidata
  collection. This returns a data frame with conditions parameters
  and measures on the polarization data like linear regression slope,
  intercept, min, max, final value of the mean values, variance, etc.

  This reports data broken up by a polarization regression slope threshold
  and thus partitions the results into polarizing, depolarizing, and
  remaining the same.

  :param multidata: A collection of multidata for a given experiment.
  '''
  slope_threshold = 0.01
  intercept_threshold = 8.5

  polarization_data = { key: value for (key,value) in multidata.items() if key[1] == 'polarization' }
  polarization_means = { key: value['0'].mean(0) for (key,value) in polarization_data.items() }
  polarization_vars = { key: value['0'].var(0).mean() for (key,value) in polarization_data.items() }
  x = np.array([[val] for val in range(len(list(polarization_means.values())[0]))])
  df = pd.DataFrame(columns=['translate','tactic','media_dist','citizen_dist','epsilon','graph_type','ba-m','repetition','lr-intercept','lr-slope','var','start','end','delta','max'])

  for (props, data) in polarization_means.items():
    model = LinearRegression().fit(x, data)
    df.loc[len(df.index)] = list(props[0]) + [model.intercept_,model.coef_[0],polarization_vars[props],data[0],data[-1],data[-1]-data[0],max(data)]
  
  polarizing = df[df['lr-slope'] >= slope_threshold]
  depolarizing = df[df['lr-slope'] <= slope_threshold*-1]
  same = df[(df['lr-slope'] > slope_threshold*-1) & (df['lr-slope'] < slope_threshold)]
  polarizing = polarizing.append(same[same['lr-intercept'] >= intercept_threshold])
  depolarizing = depolarizing.append(same[same['lr-intercept'] < intercept_threshold])

  return { 'polarizing': polarizing, 'nonpolarizing': depolarizing, 'same': same}

def polarizing_results_analysis(df):
  '''
  One specific analysis that supports Table 3 in the Rabb & Cowen
  paper on a static ecosystem cascade model. It returns the proportion
  of results within result partitions by institution tactic, when
  parameters epsilon, gamma, and h_G are set certain ways.
  '''
  tactics = ['broadcast-brain', 'appeal-mean', 'appeal-median', 'appeal-mode']
  for tactic in tactics:
    total = len(df[df['tactic'] == tactic])
    print(f'{tactic} ({total})')
    print(len(df.query(f'tactic == "{tactic}" and epsilon=="0"')) / total)
    print(len(df.query(f'tactic == "{tactic}" and epsilon=="1"')) / total)
    print(len(df.query(f'tactic == "{tactic}" and epsilon=="2"')) / total)
    print(len(df.query(f' tactic == "{tactic}" and translate=="0"')) / total)
    print(len(df.query(f'tactic == "{tactic}" and translate=="1"')) / total)
    print(len(df.query(f'tactic == "{tactic}" and translate=="2"')) / total)
    print(len(df.query(f'tactic == "{tactic}" and graph_type=="ba-homophilic"')) / total)
    print(len(df.query(f'tactic == "{tactic}" and graph_type=="barabasi-albert"')) / total)

def polarizing_results_analysis_by_param(df, params):
  '''
  Query polarization results by all parameter combinations
  and return results.

  :param df: The experiment results data frame for polarization
  results.
  :param params: A dictionary of parameters to use for
  segmentation during results analysis (currently hardcoded below).
  '''
  params = {
    'translate' : ['0', '1', '2'],
    'epsilon' : ['0', '1', '2'],
    'tactic' : ['broadcast-brain', 'appeal-mean', 'appeal-median', 'appeal-mode'],
    'media_dist' : [ 'uniform', 'normal', 'polarized' ],
    'graph_type' : [ 'ba-homophilic', 'barabasi-albert' ],
    'citizen_dist' : ['normal', 'uniform', 'polarized'],
  }

  num_params = 6
  all_params = []
  for param in params.keys():
    param_list = params[param]
    for val in param_list:
      all_params.append((param, val))

  combos = {}
  for combo_len in range(1, num_params):
    for combo in itertools.combinations(all_params, combo_len):
      unique_keys = set([pair[0] for pair in combo])
      flat_combo = ({key: [pair[1] for pair in combo if pair[0] == key] for key in unique_keys})
      combos[len(combos)] = flat_combo
    # for combo in itertools.product(*param_combos, combo_len):
    #   combos.append(combo)
  
  ratios = {}
  param_dfs = {}
  for combo_i in combos.keys():
    combo = combos[combo_i]
    query = '('
    for param in combo.keys():
      query += f'('
      for val in combo[param]:
        query += f'{param}=="{val}" or '
      query = query[:-4] + ') and '
    query = query[:-5] + ')'
    param_dfs[combo_i] = df.query(query)
    ratios[combo_i] = len(param_dfs[combo_i]) / len(df)
  
  return (combos, param_dfs, ratios)

  # return combos