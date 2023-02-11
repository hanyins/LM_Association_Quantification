import json
import pickle
import os

Google_RE_fnames = ["../lama/Google_RE/" + s for s in os.listdir("../lama/Google_RE/")]
TREx_fnames = ["../lama/TREx/" + s for s in os.listdir("../lama/TREx/")]

# for find cooccurrence 
unique_pairs = set()
total_count = 0

# relation dict
skip = ['P413', 'P136', 'P190', 'P1923', 'P102']
templates = {}
with open("../lama/relations.jsonl") as jsonlf:
    jsonl = list(jsonlf)
for jsonstr in jsonl:
    relation = json.loads(jsonstr)
    if relation['relation'] not in skip:
        templates[relation['relation']] = relation['template'].replace('[X]', '{}')[:-6]
    
templates['P106'] = "By profession {} is a"
templates['P27'] = "{} is a citizen of"
templates['P140'] = "{} is affiliated with the religion of"
    
# print(templates['P413']) # [X] plays in [Y] position .
# # print(templates['P106']) # [X] is a [Y] by profession .
# # print(templates['P27']) # [X] is [Y] citizen .
# print(templates['P136']) # [X] plays [Y] music .
# print(templates['P190']) # [X] and [Y] are twin cities .
# 'P140': '{} is affiliated with the [Y] religion'

for fname in TREx_fnames:
    relation = fname.split('/')[-1][:-6]
    if relation not in skip:
        prompts = []
        curr_template = templates[relation]
        with open(fname, 'r') as jsonf:
            jsonl = list(jsonf)
        for jsonstr in jsonl:
            data = json.loads(jsonstr)
            sub, obj = data['sub_label'], data['obj_label']
            pair = (sub, obj)
            unique_pairs.add(pair)
            prompts.append((sub, obj, curr_template.format(sub)))
            
        total_count += len(prompts)
        with open("./prompts/{}.pkl".format(relation), 'wb') as f:
            pickle.dump(prompts, f)
            
with open("TREx_pairs.pkl", 'wb') as f:
    pickle.dump(unique_pairs, f)

print(total_count)
