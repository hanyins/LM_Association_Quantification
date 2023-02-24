from files import *
from collections import defaultdict
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from transformers import GPT2Tokenizer, AutoModelForCausalLM
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# 50, 100, and 200 tokens preceding
if __name__ == "__main__":
    n_sent = 2
    max_sent = 16
    
    pairs = read_pickle("data/TREx_pairs.pkl")
    
    fnames = get_fnames_from_path("occurrence_agg")
    
    # contexts_50 = defaultdict(list)
    # contexts_100 = defaultdict(list)
    # contexts_200 = defaultdict(list)
    contexts = defaultdict(list)
    
    reconstructedSentence = TreebankWordDetokenizer()

    for _, obj in tqdm(pairs):
        # print(obj)
        for fname in fnames:
            # print(len(contexts_50[obj]))
            # if len(contexts_50[obj]) >= max_sent:
            #     break
            
            if len(contexts[obj]) >= max_sent:
                break
            
            i = fname.split('_')[-1].split(".")[0]
            wiki_fname = f"../wikidump20200301/wikidump-{i}.pkl"
            docs = read_pickle(wiki_fname)
            
            data = read_pickle(fname)
            # dict to defaultdict
            data = defaultdict(lambda: defaultdict(list), data)
            
            obj_dict = data[obj]
            doc_n = list(obj_dict.keys())[:n_sent]
            # print(doc_n)
            # print(obj)
            
            for n in doc_n:
                index = obj_dict[n][0]
                context = str(docs[n])[:index]
                # tokens = word_tokenize(context)
                contexts[obj].append(context)
                # print(context[-250:])
                # print(len(tokenizer(context)["input_ids"]))
                
                # ltokens = len(tokens)
                # tokens_50 = tokens[max(0, ltokens-50):]
                # tokens_100 = tokens[max(0, ltokens-100):]
                # tokens_200 = tokens[max(0, ltokens-200):]
                
                
                # contexts_50[obj].append(reconstructedSentence.detokenize(tokens_50))
                # contexts_100[obj].append(reconstructedSentence.detokenize(tokens_100))
                # contexts_200[obj].append(reconstructedSentence.detokenize(tokens_200))
        
            
    
    # save_pickle(contexts_50, "prompts_context/prompts_context_50.pkl")
    # save_pickle(contexts_100, "prompts_context/prompts_context_100.pkl")
    # save_pickle(contexts_200, "prompts_context/prompts_context_200.pkl")
    
    save_pickle(contexts, "prompts_context/contexts.pkl")