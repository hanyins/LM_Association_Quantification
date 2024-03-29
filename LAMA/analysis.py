import numpy as np
from collections import defaultdict
from files import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr

markers_colors = {"2.7B-greedy": ("D", "#518CD8"), 
                  "1.3B-greedy": ("H", "#FEB40B"),
                  "125M-greedy": ("p", "#6DC354"),
                  "6B-greedy": ("8", "#FD6D5A"),
                  "20B-greedy": ("d", "#9467bd")}
hist_color = "#454D66"

# span_10,span_20,span_50,span_100,span_200
def get_score(cooccurrence, sub, obj, w):
    cnt = list(cooccurrence[(sub, obj)].values())
    cnt.insert(0, 0)
    diff = [cnt[i] - cnt[i-1] for i in range(1, len(cnt))]
    return sum([x * y for (x, y) in zip(diff, w)])

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

def reject_outlier(score_chunk, acc):
    score_chunk = np.array(score_chunk)
    acc = np.array(acc)
    Q1 = np.quantile(score_chunk, 0.25)
    Q3 = np.quantile(score_chunk, 0.75)
    IQR = Q3 - Q1
    
    min_ = Q1 - 1.5 * IQR
    max_ = Q3 + 1.5 * IQR
    
    indices = [(score_chunk >= min_) & (score_chunk <= max_)]
    score_chunk = score_chunk[indices]
    acc = acc[indices]
    print(len(acc))
    
    return score_chunk, acc

def get_acc_score_equal_split(success, scores, binsize):
    scores, success = zip(*sorted(zip(scores, success)))
    scores, success = list(scores), list(success)
    
    score_chunks = [scores[i:i+binsize] for i in range(0,len(scores),binsize)]
    success_chunks = [success[i:i+binsize] for i in range(0,len(success),binsize)]
            
    # get average score of each chunk
    # score_chunks[-2].extend(score_chunks[-1])
    # score_chunks.pop()
    avg_scores = [sum(i)/len(i) for i in score_chunks]
    
    # get accuracy of each chunk
    # success_chunks[-2].extend(success_chunks[-1])
    # success_chunks.pop()
    acc = [sum(i)/len(i) for i in success_chunks]
    
    print("Last chunk size: ", len(success_chunks[-1]))
    return avg_scores, acc, score_chunks, len(acc)

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

def plot_acc_score_models(bins, models=["6B-greedy", "2.7B-greedy", "1.3B-greedy", "125M-greedy"], markers=["D", 'H', 'p'], draw_context=True):
    plt.clf()
    
    # prepare plots
    fig = plt.figure()
    
    gs = GridSpec(3,2)
    ax_joint = fig.add_subplot(gs[0:2,0:2])
    ax_marg_x = fig.add_subplot(gs[2:3,0:2])
    
    # set up axes
    ax_joint.set_xscale("log")
    ax_marg_x.set_xscale("log")
    
    ax_joint.set_xticks(bins)
    ax_marg_x.set_xticks(bins)
    ax_joint.set_axisbelow(True)
    ax_marg_x.set_axisbelow(True)
    ax_joint.grid(color="lightgray", linewidth=0.5)
    ax_marg_x.grid(color="lightgray", linewidth=0.5)
    
    
    # Turn off tick labels on marginals
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    print("Model\tSpearman\tPearson")
    for model in models:
        success, _, assc_scores = read_pred_result(f"./pred_result_base/{model}")

        avg_score, avg_acc, _ = get_acc_score_digitize(success, assc_scores, bins)
        # print(f"{model} bin=0 avg acc: {avg_acc[0]}")
        # print(f"{model} bin=1 avg acc: {avg_acc[1]}")
        # print(f"{model} bin=2 avg acc: {avg_acc[2]}")
        
        # spearman and pearson coeff. calculation
        spearman, _ = spearmanr(np.log(avg_score), avg_acc)
        pearson, _ = pearsonr(np.log(avg_score), avg_acc)
        print(f"{model}\t{spearman:.4f}\t{pearson:.4f}")
        
        marker, color = markers_colors[model]
        ax_joint.plot(avg_score, avg_acc, marker=marker, color=color, ms=3, label=f"GPT-Neo-{model}\nSpearman coeff. = {spearman:.4f}")
        
        if draw_context:
            acc_context = get_context_acc(model)
            print(f"{model} context acc: {acc_context}")
            ax_joint.axhline(acc_context, color=color, ms=2, linestyle='dashed', label=f"context-{model}")
        
    # set xlim
    xmin10, xmax10 = np.log10([bins[0], bins[-1]])
    ax_joint.set_xlim(10**(xmin10), 10**(xmax10))
    ax_marg_x.set_xlim(10**(xmin10), 10**(xmax10))
    # set ylim
    ymax1000 = 8000
    ax_marg_x.set_ylim(0,ymax1000)
    # joint plot labels and titles
    ax_joint.set_ylabel("LAMA Prediction Accuracy")
    ax_joint.legend(fontsize=6)
    # ax_joint.set_title("Accuracy vs. Score")
    
    
    # plot counts
    counts, edges, bars = ax_marg_x.hist(assc_scores, bins=bins, color=hist_color)
    ax_marg_x.bar_label(bars)
    ax_marg_x.set_ylabel("Count")
    ax_marg_x.set_xlabel("Association Easiness Scores")
    
    # global settings and save
    if draw_context:
        plt.savefig("output_figs/acc_score.jpeg",dpi=300,bbox_inches='tight')
    else:
        plt.savefig("output_figs/acc_score-nocontext.jpeg",dpi=300,bbox_inches='tight')


# Analysis 1 - Accuracy vs. Association score [Compare different model size]
if __name__ == "__main__":
    # equal chunks
    cooccurrence = defaultdict(lambda: defaultdict(int), read_pickle("data/cooccur.pkl"))
    binsize = 2000
    
    # plot_acc_score_models_equal(binsize=binsize)
    # boxplot_score_chunk("125M-greedy", binsize=binsize)
    
    # chunk by score
    bins = np.array([10**(i/2) for i in range(-3,10)])
    plot_acc_score_models(bins, draw_context=False)
    # boxplot_score_chunk("125M-greedy", equal=False, bins=bins)
    
    

    
    
    