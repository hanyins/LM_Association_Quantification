import os
import pickle
import re
import sys
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Process, Pool


def extract_function(i):
    filename = "wikidump-{}".format(str(i))
    
    with open("../wikidump20200301/{}.pkl".format(filename), 'rb') as f:
        texts = pickle.load(f)

    # name/email - filename(517350) - index
    occurrence = defaultdict(list)
    errors = defaultdict(list)
    for key in tqdm(texts.keys()):
        for label in labels:
            label = str(label)
            try:
                occur_label = [(filename, key, m.start()) for m in re.finditer(label, str(texts[key]))]
                if occur_label:
                    occurrence[label].extend(occur_label)
            except:
                errors[filename].append(key)
            
    with open("./occurrence/occur_label_{}.pkl".format(str(i)), "wb") as f:
        pickle.dump(occurrence, f)
        
    with open("./errors/occur_label_error_{}.pkl".format(str(i)), "wb") as f:
        pickle.dump(errors, f)


if __name__ == "__main__":
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    
    with open("data/lama_label_list.pkl", "rb") as f:
        labels = pickle.load(f)
    
    pool = Pool(m - n)
    
    pool.imap(extract_function, range(n, m))
    
    pool.close()
    pool.join()
    
    print(n, m)
    