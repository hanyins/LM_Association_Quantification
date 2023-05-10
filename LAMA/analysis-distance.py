import numpy as np
from collections import defaultdict
from files import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr
from brokenaxes import brokenaxes

markers_colors = {"2.7B-greedy": ("D", "#518CD8"), 
                  "1.3B-greedy": ("H", "#FEB40B"),
                  "125M-greedy": ("p", "#6DC354"),
                  "6B-greedy": ("8", "#FD6D5A"),
                  "20B-greedy": ("d", "#9467bd")}
hist_color = "#454D66"

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


def plot_acc_score_models(bins, spans=[10, 20, 50, 100, 200], models=["6B-greedy", "2.7B-greedy", "1.3B-greedy", "125M-greedy"]):
    bax = brokenaxes(ylims=((0.145, 0.2), (0.245, 0.45)), hspace=.2)
    
    for model in models:
        scores = []
        acc_list = []
        for i, span in enumerate(spans):
            weights = [0 for _ in range(5)]
            weights[i] = 1
            
            success, _, assc_scores = read_pred_result(f"./pred_result_base/{model}", w=weights)
        
            scores, success = np.array(assc_scores), np.array(success)
            success_ = success[scores>0]
            scores_ = scores[scores>0]
            avg_score, avg_acc = np.mean(scores_), np.mean(success_)
            # print(avg_score, avg_acc)
            acc_list.append(avg_acc)

        marker, color = markers_colors[model]
        bax.plot(spans, acc_list, marker=marker, color=color, ms=3, label=f"GPT-Neo-{model}")
        for x, y in zip(spans, acc_list):                                       # <--
            bax.annotate("%.5f"%y, xy=(x+1, y+0.001), textcoords='data') 
        
    
    bax.legend(fontsize=8)
    bax.set_ylabel("LAMA Prediction Accuracy")
    bax.set_xlabel("# Co-occurrence Within Distance Range")
    # plt.xscale("log")
    # plt.title("Accuracy vs. Co-occurrence Distance")
    bax.grid(color="lightgray", linewidth=0.5)
    # bax.grid(which='minor', axis='y', color='lightgray', linewidth=0.3, linestyle="--")
    

    plt.savefig("output_figs/acc_spans.jpeg", dpi=300, bbox_inches='tight')
    

# Analysis - Accuracy vs. Distance [Compare different model size]
if __name__ == "__main__":
    # equal chunks
    cooccurrence = defaultdict(lambda: defaultdict(int), read_pickle("data/cooccur.pkl"))
    binsize = 2000
    
    # chunk by score
    bins = np.array([10**(i/2) for i in range(0,10)])
    plot_acc_score_models(bins)
    