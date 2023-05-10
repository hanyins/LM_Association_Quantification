import pickle

with open("parsed_emails.pkl", "rb") as f:
    mailbodies = pickle.load(f)