import json
from tqdm import tqdm
import torch
import re
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from nltk.tokenize import word_tokenize
from files import *

gid = 1
device = f"cuda:{gid}"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def get_prompts_context(fname):
    data = read_pickle(fname)
    objs, prompts = [], []
    for obj in data:
        for prompt in data[obj]:
            objs.append(obj)
            prompts.append(prompt)
    
    return objs, prompts

def get_prompts(fname):
    # prompts: List of prompts
    tuples = read_pickle(fname)
    subs, objs, prompts = [], [], []
    for sub, obj, prompt in tuples:
        subs.append(sub)
        objs.append(obj)
        prompts.append(prompt)
    return subs, objs, prompts

if __name__ == "__main__":
    # models = ['125M', '1.3B', '2.7B']
    model_size = '2.7B'
    decoding_alg = "greedy"
    
    # modes = ["base", "shuffle", "context"]
    mode = "base"
    if mode == "shuffle":
        dir_path = "./prompts_shuffle/"
        
    elif mode == "base":
        dir_path = "./prompts/"
    
    else:
        raise Exception(f'current mode: "{mode}" not supported')
    
    print("model: gpt-neo-"+model_size)
    print("decoding:", decoding_alg)

    prompt_fnames = get_fnames_from_path(dir_path)

    model_name = f'EleutherAI/gpt-neo-{model_size}'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    bs = 16
    accuracies = {}
    
    for fname in prompt_fnames:
        subs, objs, prompts = get_prompts(fname)
        relation = fname.split('/')[-1][:-4]
            
        pred_result = []
        curr_result = []
        success_cnt = 0

        print(prompts[:3])
        
        for i in tqdm(range(0, len(prompts), bs)):
            texts = prompts[i:i+bs]
            encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
            prompt_length = (encoding["input_ids"].shape)[1]
            
            with torch.no_grad():
                if decoding_alg=="greedy":
                    generated_ids = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10, do_sample=False)
                elif decoding_alg=="top_k":
                    generated_ids = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10, do_sample=True, temperature=0.7)
                elif decoding_alg=="beam_search":
                    generated_ids = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10, num_beams=5, early_stopping=True)

                # for j,s in enumerate(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)):
                #     s = s[len(texts[j]):]
                #     pred_result.append(s)
                pred_result.extend(tokenizer.batch_decode(generated_ids[:,prompt_length:], skip_special_tokens=True))

        for sub, obj, text in zip(subs, objs, pred_result):
            found = [m.start() for m in re.finditer(obj, text)]
            if found:
                i = len(word_tokenize(text[:found[0]]))
                curr_result.append({'sub': sub, 'obj': obj, 'predicted': text, 'found': i})
                success_cnt += 1
            else:
                curr_result.append({'sub': sub, 'obj': obj, 'predicted': text, 'found': -1})
            
            
        accuracies[relation] = success_cnt / len(objs)
        with open(f"pred_result_{mode}/{model_size}-{decoding_alg}/{relation}-{model_size}-{decoding_alg}.jsonl", "w") as f:
            for item in curr_result:
                f.write(json.dumps(item) + "\n")
                
    print(accuracies)