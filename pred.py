import pickle
import json
from transformers import pipeline
from tqdm import tqdm
import torch
import re
import os
from collections import defaultdict
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import random
from nltk.tokenize import word_tokenize


gid = 1
device = f"cuda:{gid}"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


def load_pickle(filename):
    with open(filename, "rb") as pickle_handler:
        results = pickle.load(pickle_handler)
    return results

def load_csv(filename):
    results = {}
    with open(filename) as f:
        for line in f.readlines()[1:]:
            email,name = line.strip().split(',')
            results[email] = name
    return results

def get_prompts(fname):
    # prompts: List of prompts
    tuples = load_pickle(fname)
    subs, objs, prompts = [], [], []
    for sub, obj, prompt in tuples:
        subs.append(sub)
        objs.append(obj)
        prompts.append(prompt)
    return subs, objs, prompts

if __name__ == "__main__":
    models = ['125M', '1.3B', '2.7B']
    model_size = '125M'
    decoding_alg = "greedy"
    
    print("model: gpt-neo-"+model_size)
    print("decoding:", decoding_alg)
    
    dir_path = "./prompts/"
    prompt_fnames = [s for s in os.listdir(dir_path)]

    model_name = f'EleutherAI/gpt-neo-{model_size}'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    bs = 16
    accuracies = {}
    
    for fname in prompt_fnames:
        # prompts: List of prompts
        subs, objs, prompts = get_prompts(f"{dir_path}{fname}")
        relation = fname.split('.')[0]
        pred_result = []
        curr_result = []
        success_cnt = 0

        print(prompts[:3])
        
        for i in tqdm(range(0, len(prompts), bs)):
            texts = prompts[i:i+bs]
            
            encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
            with torch.no_grad():
                if decoding_alg=="greedy":
                    generated_ids = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10, do_sample=False)
                elif decoding_alg=="top_k":
                    generated_ids = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10, do_sample=True, temperature=0.7)
                elif decoding_alg=="beam_search":
                    generated_ids = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10, num_beams=5, early_stopping=True)

                for j,s in enumerate(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)):
                    s = s[len(texts[j]):]
                    pred_result.append(s)

        
        for sub, obj, text in zip(subs, objs, pred_result):
            found = [m.start() for m in re.finditer(obj, text)]
            if found:
                i = len(word_tokenize(text[:found[0]]))
                curr_result.append({'sub': sub, 'obj': obj, 'predicted': text, 'found': i})
                success_cnt += 1
            else:
                curr_result.append({'sub': sub, 'obj': obj, 'predicted': text, 'found': -1})
        
        
        accuracies[relation] = success_cnt / len(subs)
        with open(f"pred_result/{model_size}-{decoding_alg}/{relation}-{model_size}-{decoding_alg}.jsonl", "w") as f:
            for item in curr_result:
                f.write(json.dumps(item) + "\n")
                
    print(accuracies)