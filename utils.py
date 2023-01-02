import os

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