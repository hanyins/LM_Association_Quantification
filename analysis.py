import pickle
import numpy as np
from collections import defaultdict
from files import *
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
colors = ["#95a2ff", "#fa8080", "#ffc076", "#fae768", "#87e885", "#3cb9fc", "#73abf5", "#cb9bff", "#434348", "#90ed7d", "#f7a35c", "#8085e9​​​​​​​"]


# span_10,span_20,span_50,span_100,span_200
def get_score(cooccurrence, sub, obj, w=[1, 0.5, 0.2, 0.1, 0.05]):
    cnt = list(cooccurrence[(sub, obj)].values())
    cnt.insert(0, 0)
    diff = [cnt[i] - cnt[i-1] for i in range(1, len(cnt))]
    return sum([x * y for (x, y) in zip(diff, w)])

# Output
# success: list of binary 0 or 1
# results: found index (or -1 if not found)
# assc_scores: float
def read_result(path):
    success = []
    results = []
    assc_scores = []
    fnames = get_fnames_from_path(path)
    for fname  in fnames:
        data = read_jsonl(fname)
        for item in data:
            results.append(item['found'])
            if item['found'] >= 0:
                success.append(1)
            else:
                success.append(0)
            assc_scores.append(get_score(cooccurrence, item['sub'], item['obj']))
    return success, results, assc_scores

def get_acc_score_equal_split(results, scores, binsize=500):
    scores, results = zip(*sorted(zip(scores, results)))
    scores, results = list(scores), list(results)
    
    score_chunks = [scores[i:i+binsize] for i in range(0,len(scores),binsize)]
    result_chunks = [results[i:i+binsize] for i in range(0,len(results),binsize)]
    
    # get average score of each chunk
    # score_chunks[-2].extend(score_chunks[-1])
    # score_chunks.pop()
    avg_scores = [sum(i)/len(i) for i in score_chunks]
    
    # get accuracy of each chunk
    # result_chunks[-2].extend(result_chunks[-1])
    # result_chunks.pop()
    acc = [sum(i)/len(i) for i in result_chunks]
    
    print("Last chunk size: ", len(result_chunks[-1]))
    return avg_scores, acc, len(acc)

def model_size_plot(binsize=2000):
    models = ["2.7B-greedy", "1.3B-greedy", "125M-greedy"]
    markers = ["D", 'H', 'p']
    
    # start plot - compare model size
    plt.xscale("log")
    
    for model, marker in zip(models, markers):
        success, result, assc_scores = read_result(f"./pred_result/{model}")
        avg_scores, acc, n = get_acc_score_equal_split(success, assc_scores, binsize=binsize)
        
        spearman, pvalue = spearmanr(np.log(avg_scores), acc)
        pearson, ppvalue = pearsonr(np.log(avg_scores), acc)
        print(model)
        print("Spearman coeff:", spearman)
        print("P-value:", pvalue)
        print("Pearson coeff:", pearson)
        print("P-value:", ppvalue)
        
        plt.plot(avg_scores, acc, marker=marker, ms=3, label=f"GPT-Neo-{model}\nSpearman coeff. = {spearman:.4f}\nP-value = {pvalue:.4e}")
        
    plt.ylabel("LAMA Prediction Accuracy")
    plt.xlabel("Association Score")
    plt.title(f"Acc vs Score (Bin Size={binsize}, # of Bins={n})")
    plt.legend(fontsize=8)
    plt.savefig('acc_score.svg')
    

if __name__ == "__main__":
    cooccurrence = defaultdict(lambda: defaultdict(int), read_pickle("cooccur.pkl"))
    
    # Analysis 1
    model_size_plot(binsize=2000)
    
    # Analysis 2
    

    
    
    