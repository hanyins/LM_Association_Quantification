import pickle
import os
from collections import defaultdict
from tqdm import tqdm

fnames = ["./occurrence/" + s for s in os.listdir("./occurrence")]
for fname in tqdm(fnames):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        
    agg_dict = {}
    for label in data.keys():
        indices = defaultdict(list)
        for item in data[label]:
            _, n, index = item
            indices[n].append(index)
        
        agg_dict[label] = indices.copy()
        
    with open("./occurrence_agg/{}".format(fname.split('/')[-1]), "wb") as f:
        pickle.dump(agg_dict, f)