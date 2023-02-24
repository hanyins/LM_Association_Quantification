from files import *
import numpy as np

if __name__ == "__main__":
    fnames = get_fnames_from_path("pred_result_context")
    for fname in fnames:
        res = read_jsonl(fname)
        splitted = fname.split("/")[1].split("-")
        tokens = splitted[0][5:]
        model = splitted[1]
        
        found = []
        for line in res:
            found.append(line["found"])
        
        found = np.array(found)
        total = len(found)
        acc = (found >= 0).mean()
        correct = (found > 0).sum()
        print(f"tokens: {tokens}\tmodel: {model}\ttotal count: {total}\tcorrect: {correct}\tacc: {acc}")
    