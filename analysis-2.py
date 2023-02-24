from files import * 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr

def read_pred_result(path, bins):
    fnames = get_fnames_from_path(path)
    occurrence_cnt, found = [], []
    for fname in fnames:
        jsonl = read_jsonl(fname)
        for data in jsonl:
            occurrence_cnt.append(occurrence[data['sub']] + occurrence[data['obj']])
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

def plot_acc_occur_models(bins, models=["2.7B-greedy", "1.3B-greedy", "125M-greedy"], markers=["D", 'H', 'p']):
    
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
    
    ax_joint.set_xlim(10**3.5, 10**7)
    ax_marg_x.set_xlim(10**3.5, 10**7)
    ax_marg_x.set_ylim(0, 10000)
    
    # Turn off tick labels on marginals
    plt.setp(ax_joint.get_xticklabels(), visible=False)
    
    for model, marker in zip(models, markers):
        avg_cnt, avg_acc, occurrence_cnt = read_pred_result(f"./pred_result/{model}", bins=bins)
        ax_joint.plot(avg_cnt, avg_acc, marker=marker, ms=3, label=f"{model}")
        
        spearman, pvalue = spearmanr(np.log(avg_cnt)[1:-1], avg_acc[1:-1])
        pearson, ppvalue = pearsonr(np.log(avg_cnt)[1:-1], avg_acc[1:-1])
        print(model)
        print("Spearman coeff:", spearman)
        print("P-value:", pvalue)
        print("Pearson coeff:", pearson)
        print("P-value:", ppvalue)

    # joint plot labels and titles
    ax_joint.set_ylabel("LAMA Prediction Accuracy")
    ax_joint.legend(fontsize=6)
    ax_joint.set_title("Accuracy vs. Occurrence")
    
    
    #  plot counts
    counts, edges, bars = ax_marg_x.hist(occurrence_cnt, bins=bins)
    ax_marg_x.bar_label(bars)
    ax_marg_x.set_ylabel("Count")
    ax_marg_x.set_xlabel("Sub or Obj Occurrence")
    
    # global settings and save
    plt.savefig("acc_cnt_occur.svg")
    
    

if __name__ == "__main__":
      
    # Analysis 2 - Accuracy vs single count (occurrence count of sub + obj)
    bins = np.array([10**(i/2) for i in range(7,15)])
    # bins = np.array([10**(i/4) for i in range(14,29)])
    
    occurrence = read_pickle("data/occurrence.pkl")
    plot_acc_occur_models(bins=bins)
    

    
    
    