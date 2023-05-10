from files import *
from collections import defaultdict
from tqdm import tqdm

occurrence = defaultdict(int)
fnames = get_fnames_from_path("./occurrence_agg")
for fname in tqdm(fnames):
    data = read_pickle(fname)
    for label in data:
        occurrence[label] += len(data[label])
        # occurrence[label] += 1

save_pickle(occurrence, "data/occurrence_doc.pkl")