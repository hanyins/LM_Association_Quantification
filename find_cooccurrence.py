import pickle
import csv
import os
from re import L
from collections import defaultdict
from tqdm import tqdm

def cooccur_cnt(sub_idx, obj_idx, spans):
    difference = sorted([abs(a-b) for a in sub_idx for b in obj_idx])
    diff_idx, span_idx = 0, 0
    M, N = len(difference), len(spans)
    curr_span = spans[span_idx]
    # print(difference, spans)
    
    cnt = defaultdict(int)
    
    while diff_idx < M and span_idx < N:
        
        if difference[diff_idx] <= curr_span:
            diff_idx += 1
        else:
            cnt[curr_span] = diff_idx
            span_idx += 1
            if span_idx < N:
                curr_span = spans[span_idx]
            else:
                break
    
    if span_idx < N:
        for i in range(span_idx, N):
            cnt[spans[i]] = diff_idx
    
    return cnt
                
        
def get_cooccurrence(pairs, spans=[10, 20, 50, 100, 200]):
    fnames = ["./occurrence_agg/" + s for s in os.listdir("./occurrence_agg/")]
    # cnt[(sub, obj)] = # of cooccurrence
    result_cnt = defaultdict(lambda: defaultdict(int))
    
    for fname in tqdm(fnames):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
            
        # dict to defaultdict
        data = defaultdict(lambda: defaultdict(list), data)
        
        for sub, obj in pairs:
            sub_dict, obj_dict = data[sub], data[obj]
            intersection = list(set(sub_dict.keys()) & set(obj_dict.keys()))
            for doc in intersection:
                sub_idx, obj_idx = sub_dict[doc], obj_dict[doc]
                doc_cnt = cooccur_cnt(sub_idx, obj_idx, spans)
                for span in spans:
                    result_cnt[(sub, obj)][span] += doc_cnt[span]
                    # print(result_cnt)
    
    return result_cnt
                
            


if __name__ == "__main__":

    with open("data/TREx_pairs.pkl", "rb") as f:
        pairs = pickle.load(f)
    print("# of pairs:", len(pairs))
    
    cooccurrence = get_cooccurrence(pairs)
    # print(cooccurrence)
    
    # field names 
    fields = ['sub', 'obj', 'span_10', 'span_20', 'span_50', 'span_100', 'span_200'] 

    # name of csv file 
    filename = "data/cooccurrence_sub_obj.csv"
        
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        for sub, obj in pairs:
            cooccur_10 = cooccurrence[(sub, obj)][10]
            cooccur_20 = cooccurrence[(sub, obj)][20]
            cooccur_50 = cooccurrence[(sub, obj)][50]
            cooccur_100 = cooccurrence[(sub, obj)][100]
            cooccur_200 = cooccurrence[(sub, obj)][200]
            csvwriter.writerow([sub, obj, cooccur_10, cooccur_20, cooccur_50, cooccur_100, cooccur_200])
    
    # save defaultdict
    with open('data/cooccur.pkl', 'wb') as f:
        pickle.dump(dict(cooccurrence), f)
    

