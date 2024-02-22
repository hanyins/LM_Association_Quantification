# Quantifying Association Capabilities of Large Language Models
Code for [Quantifying Association Capabilities of Large Language Models and Its Implications on Privacy Leakage](https://arxiv.org/abs/2305.12707)

## USAGE
There are two parts in the experiment. The code and data is saved in `/ENRON` and `/LAMA` respectively.
### ENRON EMAIL
#### 1. **Download Dataset**
The Enron email dataset is downloaded from http://www.cs.cmu.edu/~enron/ and unzipped to `ENRON/enron/maildir/`.
#### 2. **Prepare Data**
Some data files are too large to be uploaded. Here is how you can prepare these data.
- `ENRON/enron/parsed_emails.pkl`: Run the scripts in `mailparser.ipynb`
- `ENRON/enron_count/cooccur.pkl`: `python ENRON/enron_count/cooccurrence.py`
#### 3. **Predict Results**
The prediction script is reused from this repo: https://github.com/jeffhj/LM_PersonalInfoLeak

However, some results are uploaded under `ENRON/final_result_pkl/`.

#### 4. **Analyze**
`/ENRON/analysis-email*.py` are the analysis scripts used in the experiments.

### LAMA
#### 1. **Download Dataset**
The [LAMA dataset](https://github.com/facebookresearch/LAMA) is downloaded from https://dl.fbaipublicfiles.com/LAMA/data.zip

We also used [Wikidump](https://dumps.wikimedia.org/) to extract contexts. The preprocessing script is `LAMA/wikidump-prepare.py`.

#### 2. **Prepare prompts**
- Prepare prompts: `prompt-prepare.py`
- Prepare contexts:
    1. `LAMA/find_occurrence.py`
    2. `LAMA/occurrence_agg.py`
    3. `LAMA/extract_context.py`

#### 3. **Predict Results**
- Predict: `LAMA/pred.py`
- Predict (context): `LAMA/pred_context.py`

#### 4. **Analyze**
`/LAMA/analysis*.py` are the analysis scripts used in the experiments.