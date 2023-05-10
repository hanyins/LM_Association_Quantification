import numpy as np
from collections import defaultdict
from files import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

markers_colors = {"2.7B-greedy": ("D", "#518CD8"), 
                  "1.3B-greedy": ("H", "#FEB40B"),
                  "125M-greedy": ("p", "#6DC354"),
                  "6B-greedy": ("8", "#FD6D5A"),
                  "20B-greedy": ("d", "#9467bd")}
hist_color = "#454D66"

# span_10,span_20,span_50,span_100,span_200
# def get_score(cooccurrence, sub, obj, w):
#     cnt = list(cooccurrence[(sub, obj)].values())
#     return sum([x * y for (x, y) in zip(cnt, w)])


def get_score(email, name, w=[0,0,0,0,0]):
    w = np.array(w)
    
    cooccur10 = cooccur['span=10'][(email, name)]
    cooccur20 = cooccur['span=20'][(email, name)]
    cooccur50 = cooccur['span=50'][(email, name)]
    cooccur100 = cooccur['span=100'][(email, name)]
    cooccur200 = cooccur['span=200'][(email, name)]
    
    return np.sum(np.array([cooccur10, cooccur20, cooccur50, cooccur100, cooccur200]) * w)


# Output
# success: list of binary 0 or 1
# results: found index (or -1 if not found)
# assc_scores: float
def read_pred_result(fname, w):
    with open(fname, "rb") as f:
        result = pickle.load(f)
        
    
    scores, success = [], []

    for name, pred_email in result.items():
        if name2email[name] == pred_email:
            scores.append(get_score(pred_email, name, w))
            success.append(1)
        else:
            email = name2email[name]
            scores.append(get_score(email, name, w))
            success.append(0)

    return scores, success


def get_acc_score_digitize(success, scores, bins):
    scores = np.array(scores)
    success = np.array(success)
    
    inds = np.digitize(scores, bins)
    
    avg_acc, avg_score = [], []
    score_chunks = []
    # print(scores.min(), scores.max())
    
    for nbin in range(1, len(bins)):
        avg_acc.append(np.mean(success[inds==nbin]))
        avg_score.append(np.mean(scores[inds==nbin]))
        
        score_chunks.append(scores[inds==nbin])
        
    return avg_score, avg_acc, score_chunks


def plot_acc_freq_models(bins, models=["20B-greedy", "6B-greedy", "2.7B-greedy", "1.3B-greedy", "125M-greedy"]):
    plt.clf()
    
    spans = [10, 20, 50, 100, 200]
    
    for dist in range(0,5):
        # dist = 0 -> distance <= 10
        # dist = 1 -> distance <= 20
        # dist = 2 -> distance <= 50
        # ...
    
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
        
        
        for model in models:
            weights = [0 for _ in range(5)]
            weights[dist] = 1
            
            scores, success = read_pred_result(f"./final_result_pkl/zero_shot-d-{model}.pkl", w=weights)
            # success, _, assc_scores = read_pred_result(f"./pred_result_base/{model}")

            avg_score, avg_acc, score_chunks = get_acc_score_digitize(success, scores, bins)
            # avg_score, avg_acc, _ = get_acc_score_digitize(success, assc_scores, bins)
            
            if model.startswith("20B"):
                model_name = f"GPT-NeoX-{model}"
            elif model.startswith("6B"):
                model_name = f"GPT-J-{model}"
            else:
                model_name = f"GPT-Neo-{model}"
            marker, color = markers_colors[model]
            
            ax_joint.plot(avg_score, avg_acc, marker=marker, color=color, ms=3, label=f"{model_name}")
            
        # set xlim
        xmin10, xmax10 = np.log10([bins[0], bins[-1]])
        ax_joint.set_xlim(10**(xmin10), 10**(xmax10))
        ax_marg_x.set_xlim(10**(xmin10), 10**(xmax10))
        # set ylim
        ymax1000 = 1000
        ax_marg_x.set_ylim(0,ymax1000)
        # joint plot labels and titles
        ax_joint.set_ylabel("Enron Prediction Accuracy", fontsize=12)
        ax_joint.legend(fontsize=10)
        # ax_joint.set_title("Accuracy vs. Frequency (LAMA)")
        
        
        # plot counts
        counts, edges, bars = ax_marg_x.hist(scores, bins=bins, color=hist_color)
        ax_marg_x.bar_label(bars)
        ax_marg_x.set_ylabel("Count", fontsize=12)
        ax_marg_x.set_xlabel(f"# Co-occurrence Within Distance {spans[dist]}", fontsize=12)
        
        # global settings and save
        plt.savefig(f"output_figs/acc_email_freq_{spans[dist]}.jpeg", dpi=300, bbox_inches='tight')
    

# Analysis 1 - Accuracy vs. Association score [Compare different model size]
if __name__ == "__main__":
    with open('enron/name2email.pkl', 'rb') as f:
        name2email = pickle.load(f)
    with open('enron_count/cooccur.pkl', 'rb') as f:
        cooccur = pickle.load(f)
    
    bins = np.array([10**(i/4) for i in range(0,9)])
    plot_acc_freq_models(bins)
    # boxplot_score_chunk("125M-greedy", equal=False, bins=bins)
    