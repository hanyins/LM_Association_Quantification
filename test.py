from files import *
from collections import defaultdict
from nltk.tokenize import word_tokenize
import re
import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM

# gid = 1
# device = f"cuda:{gid}"

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"

def get_prompts_context(fname):
    data = read_pickle(fname)
    objs, prompts = [], []
    for obj in data:
        for prompt in data[obj]:
            objs.append(obj)
            prompts.append(prompt)
    for line in data["archbishop"]:
        print()
        print()
        print(line)
    return objs, prompts

def extract_function(i):
    filename = "wikidump-{}".format(str(i))
    
    with open("../wikidump20200301/{}.pkl".format(filename), 'rb') as f:
        texts = pickle.load(f)

    # name/email - filename(517350) - index
    occurrence = defaultdict(list)
    errors = defaultdict(list)
    for key in texts.keys():
        for label in labels:
            label = str(label)
            try:
                occur_label = [(filename, key, m.start()) for m in re.finditer(label, str(texts[key]))]
                if occur_label:
                    occurrence[label].extend(occur_label)
            except:
                errors[filename].append(key)
                
def get_prompts_context(fname):
    data = read_pickle(fname)
    objs, prompts = [], []
    for obj in data:
        for prompt in data[obj]:
            objs.append(obj)
            prompts.append(prompt)
    
    return objs, prompts

# labels = ["archbishop"]

# occurrence = read_pickle("occurrence_test/occur_label_0.pkl")
# for fname, n, index in occurrence["archbishop"]:
#     data = read_pickle(f"../wikidump20200301/{fname}.pkl")
#     print()
#     print(str(data[n])[index-50:index])
#     print(str(data[n])[index:index+50])

# objs, prompts = get_prompts_context("prompts_context/prompts_context_200.pkl")
        
# MAX_TOKENS = 50
# texts = prompts[0:16]
# tokens = tokenizer(texts, padding=True, return_tensors='pt')
# # print(tokens["input_ids"].shape)
# tokens["input_ids"] = tokens["input_ids"][:,-MAX_TOKENS:]
# tokens["attention_mask"] = tokens["attention_mask"][:,-MAX_TOKENS:]
# print(type(tokens))

# prompts = read_pickle("prompts_context/prompts_context_50.pkl")
# prompts2 = read_pickle("prompts_context/contexts.pkl")

# print(prompts["Google"])
# print(prompts2["Google"])

fnames = get_fnames_from_path("./prompts")
count = 0
for fname in fnames:
    p = read_pickle(fname)
    count += len(p)
    
print(count)