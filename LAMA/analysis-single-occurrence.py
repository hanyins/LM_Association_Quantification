from files import * 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr

markers_colors = {"2.7B-greedy": ("D", "#518CD8"), 
                  "1.3B-greedy": ("H", "#FEB40B"),
                  "125M-greedy": ("p", "#6DC354"),
                  "6B-greedy": ("8", "#FD6D5A"),
                  "20B-greedy": ("d", "#9467bd")}
hist_color = "#454D66"

def read_pred_result(path, bins):
    fnames = get_fnames_from_path(path)
    occurrence_cnt, found = [], []
    for fname in fnames:
        jsonl = read_jsonl(fname)
        for data in jsonl:
            occurrence_cnt.append(occurrence[data['obj']])
            found.append(data['found'])
            
    occurrence_cnt = np.array(occurrence_cnt)
    success = np.where(np.array(found) >= 0, 1, 0)
    # right=False: bins[i-1] <= x < bins[i]
    inds = np.digitize(occurrence_cnt, bins)
    
    avg_acc, avg_cnt = [], []
    
    for nbin in range(1, len(bins)):
        avg_acc.append(np.mean(success[inds==nbin]))
        avg_cnt.append(np.mean(occurrence_cnt[inds==nbin]))
        
    return avg_cnt, avg_acc, occurrence_cnt

def plot_acc_occur_models(bins, models=["6B-greedy", "2.7B-greedy", "1.3B-greedy", "125M-greedy"], markers=["D", 'H', 'p']):
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
    
    ax_joint.set_xlim(10**4, 10**6.5)
    ax_joint.set_ylim(0.08,0.4)
    ax_marg_x.set_xlim(10**4, 10**6.5)
    ax_marg_x.set_ylim(0, 6000)
    
    # Turn off tick labels on marginals
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    print("Model\tSpearman\tPearson")
    for model in models:
        avg_cnt, avg_acc, occurrence_cnt = read_pred_result(f"./pred_result_base/{model}", bins=bins)
        
        spearman, _ = spearmanr(np.log(avg_cnt)[1:-1], avg_acc[1:-1])
        pearson, _ = pearsonr(np.log(avg_cnt)[1:-1], avg_acc[1:-1])
        print(f"{model}\t{spearman:.4f}\t{pearson:.4f}")
        
        marker, color = markers_colors[model]
        ax_joint.plot(avg_cnt, avg_acc, marker=marker, color=color, ms=3, label=f"GPT-Neo-{model}\nSpearman coeff. = {spearman:.4f}")

    # joint plot labels and titles
    ax_joint.set_ylabel("LAMA Prediction Accuracy")
    ax_joint.legend(fontsize=6)
    # ax_joint.set_title("Accuracy vs. Occurrence")
    
    
    #  plot counts
    counts, edges, bars = ax_marg_x.hist(occurrence_cnt, bins=bins, color=hist_color)
    ax_marg_x.bar_label(bars)
    ax_marg_x.set_ylabel("Count")
    ax_marg_x.set_xlabel("# Occurrence of Subject or Object")
    
    # global settings and save
    plt.savefig("output_figs/acc_cnt_occur.jpeg", dpi=300)
    
    

if __name__ == "__main__":
      
    # Analysis 2 - Accuracy vs single count (occurrence count of sub + obj)
    # bins = np.array([10**(i/2) for i in range(7,15)])
    bins = np.array([10**(i/4) for i in range(16,27)])
    
    occurrence = read_pickle("data/occurrence.pkl")
    plot_acc_occur_models(bins=bins)
    

    
    
    