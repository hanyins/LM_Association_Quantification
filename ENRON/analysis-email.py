import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from matplotlib.gridspec import GridSpec

markers_colors = {"2.7B-greedy": ("D", "#518CD8"), 
                  "1.3B-greedy": ("H", "#FEB40B"),
                  "125M-greedy": ("p", "#6DC354"),
                  "6B-greedy": ("8", "#FD6D5A"),
                  "20B-greedy": ("d", "#9467bd")}
hist_color = "#454D66"

# markers=["D", 'H', 'p', '8','d']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# c_string{1} = {'#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295'};

# c_string{2} = {'#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#253777'};

# c_string{3} = {'#C1C976', '#C8A9A1', '#FEC2E4', '#77CCE0', '#FFD372', '#F88078'};

# c_string{4} = {'#104FFF', '#2FD151', '#64C7B8', '#FF1038', '#45CAFF', '#B913FF'};

# c_string{5} = {'#4C87D6', '#F38562', '#F2B825', '#D4C114', '#88B421', '#199FE0'};

# c_string{6} = {'#037CD2', '#00AAAA', '#927FD3', '#E54E5D', '#EAA700', '#F57F4B'};

# c_string{7} = {'#64B6EA', '#FB8857', '#A788EB', '#80D172', '#FC7A77', '#61D4D5'};

# c_string{8} = {'#F1787D', '#F8D889', '#69CDE0', '#5EB7F1', '#EDA462', '#F6C4E6'};

# c_string{9} = {'#8C8FD5', '#C0E5BC', '#8C8FD5', '#BDF4FC', '#C3BCE6', '#F48FB1'};


# cooccurence * weight
def get_association_strength(email, name, weight=[1, 0.5, 0.25, 0.125, 0.05]):
    weight = np.array(weight)
    
    cooccur10 = cooccur['span=10'][(email, name)]
    cooccur20 = cooccur['span=20'][(email, name)]
    cooccur50 = cooccur['span=50'][(email, name)]
    cooccur100 = cooccur['span=100'][(email, name)]
    cooccur200 = cooccur['span=200'][(email, name)]
    
    return np.sum(np.array([cooccur10, cooccur20-cooccur10, cooccur50-cooccur20, cooccur100-cooccur50, cooccur200-cooccur100]) * weight)

def read_pred_result(fname):
    with open(fname, "rb") as f:
        result = pickle.load(f)
        
    
    scores, success = [], []

    for name, pred_email in result.items():
        if name2email[name] == pred_email:
            scores.append(get_association_strength(pred_email, name))
            success.append(1)
        else:
            email = name2email[name]
            scores.append(get_association_strength(email, name))
            success.append(0)
    
    # print(np.sum(np.array(scores) > 0.1))
    # print(fname)
    # print(len(scores))
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
    
    # print("Model\tSpearman\tPearson")
    for model in models:
        assc_scores, success = read_pred_result(f"./final_result_pkl/{model}")
        # print(min(assc_scores), max(assc_scores))
        scores.append(assc_scores)
        avg_score, avg_acc, _ = get_acc_score_digitize(success, assc_scores, bins)
        
        spearman, pvalue = spearmanr(np.log(avg_score), avg_acc)
        pearson, ppvalue = pearsonr(np.log(avg_score), avg_acc)
        
        # print(f"{model}\t{spearman:.4f}\t{pearson:.4f}")
        # print(model)
        # print("Spearman coeff:", spearman)
        # print("Pearson coeff:", pearson)
        model_ = model.split("-d-")[-1][:-4]
        if model_.startswith("20B"):
            model_name = f"GPT-NeoX-{model_}"
        elif model_.startswith("6B"):
            model_name = f"GPT-J-{model_}"
        else:
            model_name = f"GPT-Neo-{model_}"
        
        marker, color = markers_colors[model_]
        print(model)
        print(avg_score, avg_acc)
        ax_joint.plot(avg_score, avg_acc, marker=marker, color=color, ms=3, label=f"{model_name}\nSpearman coeff. = {spearman:.4f}")

    ymax1000 = np.ceil(max(assc_scores) / 1000) * 1000
    ax_marg_x.set_ylim(0,ymax1000)
    # joint plot labels and titles
    ax_joint.set_ylabel("Enron Prediction Accuracy")
    ax_joint.legend(fontsize=6)
    # ax_joint.set_title("Accuracy vs. Score (email)")
    
    
    # plot counts
    counts, edges, bars = ax_marg_x.hist(scores[0], bins=bins, color=hist_color, label=models)
    ax_marg_x.bar_label(bars)
    ax_marg_x.set_ylabel("Count")
    ax_marg_x.set_xlabel("Association Easiness Score")
    
    # global settings and save
    plt.savefig("output_figs/acc_score_email.jpg",dpi=300,bbox_inches='tight')

if __name__ == "__main__":
    
    with open('enron/name2email.pkl', 'rb') as f:
        name2email = pickle.load(f)
    with open('enron_count/cooccur.pkl', 'rb') as f:
        cooccur = pickle.load(f)
        
    # print(cooccur.keys())
    
    bins = np.array([10**(i/2) for i in range(-3,6)])
    plot_acc_score_models(bins)

