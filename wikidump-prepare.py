import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
from tqdm import tqdm

MAX_LEN = 23567
ds = tfds.load("wikipedia/20200301.en", split='train')

curr_count = 0
file_count = 0
curr_texts = {}
for example in tqdm(ds):
    title = example['title'].numpy().decode("utf-8")
    body = example['text'].numpy().decode("utf-8")
    text = "{}\n\n{}".format(title, body)
    curr_texts[curr_count] = text
    
    curr_count += 1
    if curr_count >= MAX_LEN:
        with open("../wikidumpcnt20200301/wikidump-{}.pkl".format(file_count), "wb") as f:
            pickle.dump(curr_texts, f)
        curr_texts = {}
        curr_count = 0
        file_count += 1
    
    
with open("../wikidumpcnt20200301/wikidump-{}.pkl".format(file_count), "wb") as f:
    pickle.dump(curr_texts, f)