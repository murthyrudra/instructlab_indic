import json
from json_repair import repair_json
from tqdm import tqdm
from dotenv import load_dotenv

from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
# GENAI_API=<genai-api-endpoint>
load_dotenv()
client = Client(credentials=Credentials.from_env())

with open("utils/ncert_taxonomy.json", "r", errors="ignore", encoding="utf8") as reader:
    taxonomy = json.load(reader)

master_prompt = """You are a teacher who teaches {subject}. You are assigned the job of creating question-answer solutions. You should first generate the question, followed by step-by-step reasoning and then the answer.
The output should be in JSON format. Each JSON entry should contain two fields: question and answer.
"""

prompts = []
for root_node in taxonomy:
    if isinstance(taxonomy[root_node], dict):
        for stem_node in taxonomy[root_node]:
            for each_subject in taxonomy[root_node][stem_node]:
                prompt = master_prompt.replace("{subject}", each_subject)
                prompts.append(prompt)
    else:
        for each_subject in taxonomy[root_node]:
            theme = f"{each_subject}"
            prompt = master_prompt.replace("{subject}", theme)
            prompts.append(prompt)

print(f"Read {len(prompts)} number of prompts")
domains = {}

# yields batch of results that are produced asynchronously and in parallel
for response in tqdm(client.text.generation.create(
    model_id="mistralai/mixtral-8x7b-instruct-v01",
    inputs=prompts,
    parameters=TextGenerationParameters(
        max_new_tokens=2048,
        min_new_tokens=512,
        return_options=TextGenerationReturnOptions(
            input_text=True,
        ),
        decoding_method="greedy",
        temperature=0.0,
    ),
)):
    result = response.results[0]

    domain = result.input_text.split("You are a teacher who teaches ")[-1].split(". You are assigned the job of creating question-answer solutions.")[0]

    if domain not in domains:
        domains[domain] = []

    output = result.generated_text.strip()
    output = json.loads(repair_json(output))

    import pdb
    pdb.set_trace()

    try:
        if isinstance(output, list):
            for each_qa_pair in output:
                if "question" in each_qa_pair and "answer" in each_qa_pair:
                    domains[domain].append(each_qa_pair)
        else:
            for each_qa_pair in output["questions"]:
                if "question" in each_qa_pair and "answer" in each_qa_pair:
                    domains[domain].append(each_qa_pair)
    except:
        pass

    with open("utils/generated_ncert_questions.json", "w", errors="ignore", encoding="utf8") as writer:
        json.dump(domains, writer, indent = 6)

# https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-BNB-NF4-BF16