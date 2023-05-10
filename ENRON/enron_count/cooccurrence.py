import pickle
import csv
from re import L
from collections import defaultdict

# span unit: characters
def get_cooccurrence(email2name, email_occurs, name_occurs, span=100):
    cooccur = defaultdict(int)
    for email, name in email2name.items():
        email_list = email_occurs[email]
        name_list = name_occurs[name]
        count = 0
        for email_path, email_idx in email_list:
            for name_path, name_idx in name_list:
                if email_path == name_path:
                    idx_dist = min(abs(name_idx + len(name) - email_idx), abs(email_idx + len(email) - name_idx))
                    if idx_dist <= span:
                        count += 1
        cooccur[(email, name)] = count
    return cooccur


if __name__ == "__main__":

    # email, name pairs
    with open("email2name.pkl", "rb") as f:
        email2name = pickle.load(f)

    # email: (fpath, index)
    with open("occur_email.pkl", "rb") as f:
        email_occurs = defaultdict(list, pickle.load(f))

    # name: (fpath, index)
    with open("occur_name.pkl", "rb") as f:
        name_occurs = defaultdict(list, pickle.load(f))

    cooccur_10 = get_cooccurrence(email2name, email_occurs, name_occurs, span=10)
    cooccur_20 = get_cooccurrence(email2name, email_occurs, name_occurs, span=20)
    cooccur_50 = get_cooccurrence(email2name, email_occurs, name_occurs, span=50)
    cooccur_100 = get_cooccurrence(email2name, email_occurs, name_occurs, span=100)
    cooccur_200 = get_cooccurrence(email2name, email_occurs, name_occurs, span=200)

    cooccur = {'span=10': cooccur_10, 'span=20': cooccur_20, 'span=50': cooccur_50, 'span=100': cooccur_100, 'span=200': cooccur_200}
    with open('cooccur.pkl', 'wb') as f:
        pickle.dump(cooccur, f)
    
    # field names 
    fields = ['email', 'name', 'span_10', 'span_20', 'span_50', 'span_100', 'span_200'] 

    # name of csv file 
    filename = "cooccurrence_email_name.csv"
        
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        for email, name in email2name.items():
            csvwriter.writerow([email, name, cooccur_10[(email, name)], cooccur_20[(email, name)], cooccur_50[(email, name)], cooccur_100[(email, name)], cooccur_200[(email, name)]])
    
    

