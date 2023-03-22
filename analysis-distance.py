import numpy as np
from collections import defaultdict
from files import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# span_10,span_20,span_50,span_100,span_200
def get_score(cooccurrence, sub, obj, w):
    cnt = list(cooccurrence[(sub, obj)].values())
    return sum([x * y for (x, y) in zip(cnt, w)])

# Output
# success: list of binary 0 or 1
# results: found index (or -1 if not found)
# assc_scores: float
def read_pred_result(path, w=[1, 0.5, 0.25, 0.125, 0.05]):
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
            assc_scores.append(get_score(cooccurrence, item['sub'], item['obj'], w))
    return success, results, assc_scores

def get_acc_score_digitize(success, scores, bins):
    scores = np.array(scores)
    success = np.array(success)
    
    inds = np.digitize(scores, bins)
    
    avg_acc, avg_score = [], []
    score_chunks = []
    
    for nbin in range(1, len(bins)):
        avg_acc.append(np.mean(success[inds==nbin]))
        avg_score.append(np.mean(scores[inds==nbin]))
        
        score_chunks.append(scores[inds==nbin])
        
    return avg_score, avg_acc, score_chunks

def get_zero_cooccur_acc(success, scores):
    # print(np.sum(np.array(scores) == 0))
    return np.mean(np.array(success)[np.array(scores) == 0])

def get_context_acc(model):
    fname = f"pred_result_context/token200-{model}.jsonl"
    res = read_jsonl(fname)
    found = []
    for line in res:
        found.append(line["found"])
    return (np.array(found) >= 0).mean()

def plot_acc_score_models(bins, spans=[10, 20, 50, 100, 200], models=["2.7B-greedy", "1.3B-greedy", "125M-greedy"], markers=["D", 'H', 'p']):
    for model in models:
        scores = []
        
        plt.clf()
        acc_list = []
        for i, span in enumerate(spans):
            weights = [0 for _ in range(5)]
            weights[i] = 1
            
            success, _, assc_scores = read_pred_result(f"./pred_result_base/{model}", w=weights)
        
            scores, success = np.array(assc_scores), np.array(success)
            success_ = success[scores>0]
            scores_ = scores[scores>0]
            avg_score, avg_acc = np.mean(scores_), np.mean(success_)
            print(avg_score, avg_acc)
            acc_list.append(avg_acc)

        plt.plot(spans, acc_list, marker="H", ms=3)
        for xy in zip(spans, acc_list):                                       # <--
            plt.annotate("%.5f"%xy[1], xy=xy, textcoords='data') 
        
        
    plt.ylabel("LAMA Prediction Accuracy")
    plt.xlabel("# co-occurrence within distance range")
    # plt.xscale("log")
    # plt.title("Accuracy vs. Co-occurrence Distance")
    plt.grid(color="lightgray", linewidth=0.5)
    plt.grid(which='minor', axis='y', color='lightgray', linewidth=0.3, linestyle="--")
    

    plt.savefig("output_figs/acc_spans.jpeg", dpi=150)
    

# Analysis 1 - Accuracy vs. Association score [Compare different model size]
if __name__ == "__main__":
    # equal chunks
    cooccurrence = defaultdict(lambda: defaultdict(int), read_pickle("data/cooccur.pkl"))
    binsize = 2000
    
    # chunk by score
    bins = np.array([10**(i/2) for i in range(0,10)])
    plot_acc_score_models(bins)
    