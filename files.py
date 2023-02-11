import os
import json
import pickle

# file helper functions

# input: directory path
# output: a list of file names
def get_fnames_from_path(path):
    if path[-1] != "/":
        path += "/"
    fnames = [path + s for s in os.listdir(path)]
    return fnames

# input: jsonl file path
# output: a list of dicts
def read_jsonl(fname):
    with open(fname, 'r') as json_file:
        json_list = list(json_file)

    result = []
    for json_str in json_list:
        data = json.loads(json_str)
        result.append(data)
        
    return result


# input: pkl file path
# output: any type data in the pkl file
def read_pickle(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data
    