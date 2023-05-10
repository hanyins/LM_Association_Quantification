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
tokenizer.truncation_side = "left"

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
    mode = "context"
    
    print("model: gpt-neo-"+model_size)
    print("decoding:", decoding_alg)
    
    model_name = f'EleutherAI/gpt-neo-{model_size}'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    MAX_TOKENS = [50, 100, 200]
    bs = 16
    accuracies = {}
    
    fname = "prompts_context/contexts.pkl"
    
    objs, prompts = get_prompts_context(fname)
    print(len(objs), len(prompts))

    # print(prompts[:3])
    
    for MAX_TOKEN in MAX_TOKENS:
        print("MAX_TOKEN:", MAX_TOKEN)
        success_cnt = 0
        pred_result = []
        curr_result = []
        
        for i in tqdm(range(0, len(prompts), bs)):
            texts = prompts[i:i+bs]
            encoding = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=MAX_TOKEN).to(device)
            prompt_length = (encoding["input_ids"].shape)[1]
            # print(objs[i])
            # print(tokenizer.batch_decode(encoding["input_ids"], skip_special_tokens=True))
            # if prompt_length != MAX_TOKEN:
            #     print(prompt_length)
            # print(encoding)
            with torch.no_grad():
                if decoding_alg=="greedy":
                    generated_ids = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10, do_sample=False)
                elif decoding_alg=="top_k":
                    generated_ids = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10, do_sample=True, temperature=0.7)
                elif decoding_alg=="beam_search":
                    generated_ids = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10, num_beams=5, early_stopping=True)

                pred_result.extend(tokenizer.batch_decode(generated_ids[:,prompt_length:], skip_special_tokens=True))

        for obj, text in zip(objs, pred_result):
            found = [m.start() for m in re.finditer(obj, text)]
            if found:
                i = len(word_tokenize(text[:found[0]]))
                curr_result.append({'obj': obj, 'predicted': text, 'found': i})
                success_cnt += 1
            else:
                curr_result.append({'obj': obj, 'predicted': text, 'found': -1})
        
        print("pred_result length:",len(pred_result))
        accuracies[MAX_TOKEN] = success_cnt / len(pred_result)
        with open(f"pred_result_{mode}/token{MAX_TOKEN}-{model_size}-{decoding_alg}.jsonl", "w") as f:
            for item in curr_result:
                f.write(json.dumps(item) + "\n")
                
    print(accuracies)