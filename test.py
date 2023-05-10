from files import *
from collections import defaultdict

if __name__ == "__main__":
    test = "ENRON"
    # test = "LAMA"
    
    dirs = ["test-result-osprey1", "test-result-osprey2"]
    
    if test == "ENRON":
        name2email = read_pickle("ENRON/enron/name2email.pkl")
        name2email = defaultdict(str, name2email)
        for dirr in dirs:
            fnames = get_fnames_from_path(dirr)
            fnames.sort()
            for fname in fnames:
                if fname.endswith(".pkl"):
                    result = read_pickle(fname)
                    count = 0
                    for name in result:
                        if name2email[name] == result[name]:
                            count += 1
                    print(f"{fname}: {count}")
        
    elif test == "LAMA":
        for dirr in dirs:
            dirnames = get_fnames_from_path(dirr)
            for dirname in dirnames:
                if dirname.endswith("greedy"):
                    fnames = get_fnames_from_path(dirname)
                    for fname in fnames:
                        result = read_jsonl(fname)
                        count = 0
                        for item in result:
                            if item["found"] >= 0:
                                count += 1
                        print(f"{fname}: {count}")
    