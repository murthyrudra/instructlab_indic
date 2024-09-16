import pandas as pd
import ast
from tqdm import tqdm

dataset = pd.read_csv("/dccstor/cssblr/rmurthyv/IBM/InstructLab/instructlab/utils/queries_with_docs.tsv", delimiter="\t")
print(dataset)

documents = []

for index, row in tqdm(dataset.iterrows()):
    document = ast.literal_eval(row['docs'])
    documents.extend(document)
    
print(f"Read {len(documents)} number of documents")
documents = list(set(documents))
print(f"Read {len(documents)} number of documents")

for index in range(len(documents)):
    with open(f"/dccstor/cssblr/rmurthyv/IBM/InstructLab/instructlab/ncert_books/{index}.md", "w", errors="ignore", encoding="utf8") as writer:
        writer.write(documents[index])
        writer.close()