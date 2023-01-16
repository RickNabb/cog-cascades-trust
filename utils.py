import os
import math

'''
Generic utilities file to keep track of useful functions.
'''

def dict_sort(d, reverse=False):
  return {key: value for key, value in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % tuple(rgb)

def create_nested_dirs(path):
  path_thus_far = path.split('/')[0]
  for d in path.split('/')[1:]:
    if not os.path.isdir(f'{path_thus_far}/{d}'):
      # print(f'Creating {path_thus_far}/{d}')
      os.mkdir(f'{path_thus_far}/{d}')
    path_thus_far += f'/{d}'

def curr_sigmoid_p(exponent, translation):
  '''
  A curried sigmoid function used to calculate probabilty of belief
  given a certain distance. This way, it is initialized to use exponent
  and translation, and can return a function that can be vectorized to
  apply with one param -- message_distance.

  :param exponent: An exponent factor in the sigmoid function.
  :param translation: A translation factor in the sigmoid function.
  '''
  return lambda message_distance: (1 / (1 + math.exp(exponent * (message_distance - translation))))

def sigmoid_contagion_p(message_distance, exponent, translation):
  '''
  A sigmoid function to calcluate probability of belief in a given distance
  between beliefs, governed by a few parameters.
  '''
  return (1 / (1 + math.exp(exponent * (message_distance - translation))))