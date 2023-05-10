import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from matplotlib.gridspec import GridSpec
from files import *
from collections import defaultdict


markers_colors = {"2.7B-greedy": ("D", "#518CD8"), 
                  "1.3B-greedy": ("H", "#FEB40B"),
                  "125M-greedy": ("p", "#6DC354"),
                  "6B-greedy": ("8", "#FD6D5A"),
                  "20B-greedy": ("d", "#9467bd")}
hist_color = "#454D66"

# cooccurence * weight
def get_occur(name, email):
    # print(len(occur_email[email]))
    # print(len(occur_name[name]))
    # return len(occur_email[email]) + len(occur_name[name])
    return len(occur_email[email])

def read_pred_result(fname):
    with open(fname, "rb") as f:
        result = pickle.load(f)
    scores, success = [], []

    for name, pred_email in result.items():
        if name2email[name] == pred_email:
            scores.append(get_occur(name, pred_email))
            success.append(1)
        else:
            email = name2email[name]
            scores.append(get_occur(name, email))
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

def plot_acc_score_models(bins, models=["zero_shot-d-20B-greedy.pkl", "zero_shot-d-6B-greedy.pkl", "zero_shot-d-2.7B-greedy.pkl", "zero_shot-d-1.3B-greedy.pkl", "zero_shot-d-125M-greedy.pkl"], markers=["D", 'H', 'p', '8','d']):
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
    
    xmin10, xmax10 = np.log10([bins[0], bins[-1]])
    
    ax_joint.set_xlim(10**(xmin10), 10**(xmax10))
    ax_marg_x.set_xlim(10**(xmin10), 10**(xmax10))
    
    
    # Turn off tick labels on marginals
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    scores = []
    
    print("Model\tSpearman\tPearson")
    for model in models:
        assc_scores, success = read_pred_result(f"./final_result_pkl/{model}")
        # print(min(assc_scores), max(assc_scores))
        scores.append(assc_scores)
        avg_score, avg_acc, _ = get_acc_score_digitize(success, assc_scores, bins)
        
        # print(avg_score)
        # print(avg_acc)
        spearman, pvalue = spearmanr(np.log(avg_score), avg_acc)
        pearson, _ = pearsonr(np.log(avg_score), avg_acc)
        print(f"{model}\t{spearman:.4f}\t{pearson:.4f}")
        
        model_ = model.split("-d-")[-1][:-4]
        if model_.startswith("20B"):
            model_name = f"GPT-NeoX-{model_}"
        elif model_.startswith("6B"):
            model_name = f"GPT-J-{model_}"
        else:
            model_name = f"GPT-Neo-{model_}"
        
        marker, color = markers_colors[model_]
        
        ax_joint.plot(avg_score, avg_acc, marker=marker, color=color, ms=3, label=f"{model_name}\nSpearman coeff. = {spearman:.4f}")

  
    ax_marg_x.set_ylim(0,1600)
    # joint plot labels and titles
    ax_joint.set_ylabel("Enron Prediction Accuracy")
    ax_joint.legend(fontsize=6)
    # ax_joint.set_title("Accuracy vs. Score (email)")
    
    
    # plot counts
    counts, edges, bars = ax_marg_x.hist(scores[0], bins=bins, color=hist_color, label=models)
    ax_marg_x.bar_label(bars)
    ax_marg_x.set_ylabel("Count")
    ax_marg_x.set_xlabel("# Occurrence of Name or Email")
    
    # global settings and save
    plt.savefig("output_figs/acc_email_single_count.jpg",dpi=300)

if __name__ == "__main__":
    
    name2email = read_pickle("enron/name2email.pkl")
    occur_email = defaultdict(list, read_pickle("enron/occur_email.pkl"))
    occur_name = defaultdict(list, read_pickle("enron/occur_name.pkl"))
        
    bins = np.array([10**(i/2) for i in range(0,7)])
    plot_acc_score_models(bins)

