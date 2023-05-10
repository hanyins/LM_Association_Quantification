import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes

markers_colors = {"2.7B-greedy": ("D", "#518CD8"), 
                  "1.3B-greedy": ("H", "#FEB40B"),
                  "125M-greedy": ("p", "#6DC354"),
                  "6B-greedy": ("8", "#FD6D5A"),
                  "20B-greedy": ("d", "#9467bd")}
hist_color = "#454D66"

# cooccurence * weight
def get_association_strength(email, name, weight=[0,0,0,0,0]):
    weight = np.array(weight)
    
    cooccur10 = cooccur['span=10'][(email, name)]
    cooccur20 = cooccur['span=20'][(email, name)]
    cooccur50 = cooccur['span=50'][(email, name)]
    cooccur100 = cooccur['span=100'][(email, name)]
    cooccur200 = cooccur['span=200'][(email, name)]
    
    return np.sum(np.array([cooccur10, cooccur20, cooccur50, cooccur100, cooccur200]) * weight)

def read_pred_result(fname, weights):
    with open(fname, "rb") as f:
        result = pickle.load(f)
        
    scores, success = [], []

    for name, pred_email in result.items():
        if name2email[name] == pred_email:
            scores.append(get_association_strength(pred_email, name, weight=weights))
            success.append(1)
        else:
            email = name2email[name]
            scores.append(get_association_strength(email, name, weight=weights))
            success.append(0)
     
    return scores, success

# "zero_shot-d-1.3B-greedy.pkl", "zero_shot-d-125M-greedy.pkl", 
def plot_acc_score_models(bins, 
                          spans=[10, 20, 50, 100, 200], 
                          models=["zero_shot-d-20B-greedy.pkl", "zero_shot-d-6B-greedy.pkl", "zero_shot-d-2.7B-greedy.pkl"], 
                          ylims=((0.015, 0.021), (0.025, 0.035), (0.04, 0.055)),
                          fname="acc_dist_email"):

    bax = brokenaxes(ylims=ylims, hspace=.2)
    # bax = brokenaxes(ylims=((0, 0.01), (0.015, 0.021)), hspace=.2)
    
    for model in models:
        scores = []
        acc_list = []
        
        model_ = model.split("-d-")[-1][:-4]
        if model_.startswith("20B"):
            model_name = f"GPT-NeoX-{model_}"
        elif model_.startswith("6B"):
            model_name = f"GPT-J-{model_}"
        else:
            model_name = f"GPT-Neo-{model_}"
            
        marker, color = markers_colors[model_]
            
        for i, span in enumerate(spans):
            weights = [0 for _ in range(5)]
            weights[i] = 1
            
            assc_scores, success = read_pred_result(f"./final_result_pkl/{model}", weights=weights)
        
            scores, success = np.array(assc_scores), np.array(success)
            success_ = success[scores>0]
            scores_ = scores[scores>0]
            avg_score, avg_acc = np.mean(scores_), np.mean(success_)
            # print(avg_score, avg_acc)
            acc_list.append(avg_acc)
        
        # model_name = model.split("-d-")[-1][:-4]
        bax.plot(spans, acc_list, marker=marker, color=color, ms=3, label=f"{model_name}")
        for x, y in zip(spans, acc_list):
            if x == 10 and y < 0.01:
                bax.annotate("%.5f"%y, xy=(x, y-0.0008), textcoords='data') 
            else:
                bax.annotate("%.5f"%y, xy=(x+1, y+0.0001), textcoords='data') 

    
    bax.grid(color="lightgray", linewidth=0.5)
    # xmin10, xmax10 = np.log10([bins[0], bins[-1]])
    # plt.xlim(10**(xmin10), 10**(xmax10))
    
    # joint plot labels and titles
    bax.set_ylabel("Enron Prediction Accuracy",labelpad=40)
    bax.set_xlabel("# Co-occurrence Within Distance Range", labelpad=18)
    bax.legend(fontsize=8)
    # bax.legend(fontsize=6)
    # ax_joint.set_title("Accuracy vs. Score (email)")
    
    
    # global settings and save
    plt.savefig(f"output_figs/{fname}.jpg",dpi=300,bbox_inches='tight')

if __name__ == "__main__":
    
    with open('enron/name2email.pkl', 'rb') as f:
        name2email = pickle.load(f)
    with open('enron_count/cooccur.pkl', 'rb') as f:
        cooccur = pickle.load(f)
        
    # print(cooccur.keys())
    
    bins = np.array([10**(i/2) for i in range(-3,6)])
    # plot_acc_score_models(bins, 
    #                       models=["zero_shot-d-2.7B-greedy.pkl", "zero_shot-d-1.3B-greedy.pkl", "zero_shot-d-125M-greedy.pkl"], 
    #                       ylims=((0, 0.01), (0.015, 0.021)),
    #                       fname="acc_dist_email_small_models")
    plot_acc_score_models(bins)
