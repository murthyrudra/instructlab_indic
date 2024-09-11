import os
import json
from tqdm import tqdm
from datasets import load_dataset

dataset = load_dataset("ai4bharat/sangraha", "verified/hin") 

print(dataset)

pdf_dataset = dataset.filter(lambda example: example["type"].startswith("pdf"))
print(pdf_dataset)

pdf_dataset = pdf_dataset["train"]

num_docs = 0
with open("ncert_docs.jsonl", "w", errors="ignore", encoding="utf8") as writer:
    for each_doc in pdf_dataset:
        example = {}
        example["doc_id"] = num_docs
        example["docs"] = each_doc["text"]
        json.dump(example, writer, ensure_ascii=False)
        writer.write("\n")
        num_docs = num_docs + 1
        print(f"Wrote {num_docs} number of documents")
