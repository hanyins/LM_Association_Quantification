import mailparser
import os
import pickle
import re
from collections import defaultdict
from tqdm import tqdm

# read in
with open("parsed_emails.pkl", "rb") as f:
    mailbodies = pickle.load(f)

with open("email2name.pkl", "rb") as f:
    email2name = pickle.load(f)
    
with open("name2email.pkl", "rb") as f:
    name2email = pickle.load(f)


# name/email - filename - index
occurrence_name = defaultdict(list)
occurrence_email = defaultdict(list)
for filename, body in tqdm(list(mailbodies.items())[:10]):
    for email, name in email2name.items():
        occur_name = [(filename, m.start()) for m in re.finditer(name, body)]
        occur_email = [(filename, m.start()) for m in re.finditer(email, body)]
        if occur_name and len(occur_name) < 10:
            occurrence_name[name].extend(occur_name)
        if occur_email and len(occur_name) < 10:
            occurrence_email[email].extend(occur_email)

with open("occur_email.pkl", "wb") as f:
    pickle.dump(occurrence_email, f)

with open("occur_name.pkl", "wb") as f:
    pickle.dump(occurrence_name, f)