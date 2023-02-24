import random
from files import *

random.seed(8)

Google_RE_fnames = get_fnames_from_path("../lama/Google_RE")
TREx_fnames = get_fnames_from_path("../lama/TREx")

# for find cooccurrence
total_count = 0

# relation dict
# skip 5 relations and rewrite 3 relations
skip = ['P413', 'P136', 'P190', 'P1923', 'P102']
templates = {}

relations = read_jsonl("../lama/relations.jsonl")
for relation in relations:
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
        subs, objs = [], []
        curr_template = templates[relation]
        
        
        jsonl = read_jsonl(fname)
        for data in jsonl:
            sub, obj = data['sub_label'], data['obj_label']
            subs.append(sub)
            objs.append(obj)
            
        random.shuffle(subs)
        for sub, obj in zip(subs, objs):
            prompts.append((sub, obj, curr_template.format(sub)))
            
        total_count += len(prompts)
        save_pickle(prompts, "./prompts_shuffle/{}.pkl".format(relation))

print(total_count)
