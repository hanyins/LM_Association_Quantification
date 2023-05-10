label: subject or object word
n: the serial number of the article
index: the index of the label occurrence

### wikidump20200301/ (.pkl): Dict[n] = text

### occurrence.pkl: Dict[label] = count
### occurrence_doc.pkl: Dict[label] = doc count of the label (count 1 if the label appeared in one doc)
### occurrence/ (.pkl): Dict[label] = List ((filename, n, index))

### occurrence_agg/ (.pkl): Dict[label][n] = List (index)

### cooccurrence (.pkl .csv): Dict[(sub, obj)][span] = cnt

### prompts (.pkl): List (sub, obj, prompt)

### prompts_context (.pkl): Dict[obj] = List(prompt)

### pred_result (.jsonl): Dict (Keys="sub", "obj", "predicted", "found")
found == -1 if not found otherwise first index.

e.g., {"sub": "Eibenstock", "obj": "Germany", "predicted": " the heart of the city of Munich, Germany.", "found": 1}

