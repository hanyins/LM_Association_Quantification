import json
import pickle
import os

fnames = ["../lama/Google_RE/" + s for s in os.listdir("../lama/Google_RE/")]
TREx = ["../lama/TREx/" + s for s in os.listdir("../lama/TREx/")]
fnames.extend(TREx)

uniques = set()
for fname in fnames:
    with open(fname, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        uniques.add(result['sub_label'])
        uniques.add(result['obj_label'])
        
print(len(uniques))
with open("data/lama_label_list.pkl", "wb") as f:
    pickle.dump(list(uniques), f)