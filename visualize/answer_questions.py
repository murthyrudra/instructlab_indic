from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_text(obj):
    return f"""\
<|system|>
<|user|>
{obj}
<|assistant|>"""


def format_text_prompt(obj):
    return f"""\
<|system|>
"I am, Hindi Instruct Model based on Sarvam 2B, an AI language model developed by Sarvam and fine-tuned by IBM Research, based onlama-3.1 70B Instruct language model. My primary function is to be able to answer Physics based questions."
<|user|>
{obj}
<|assistant|>"""


models = [
    "meta-llama/Llama-3.2-1B",
    "sarvamai/sarvam-2b-v0.5",
    "meta-llama/Llama-3.1-8B",
]
models = models + glob("/dccstor/lm4si/rudra/models/llama_3.2_1B_instructlab/hf_step_*")
models = models + glob(
    "/dccstor/lm4si/rudra/models/sarvam-2b-v0.5_instructlab/hf_step_*"
)

combined_models = glob(
    "/dccstor/lm4si/rudra/models/llama_3.2_1B_instructlab_combined/hf_step_*"
)
combined_models = combined_models + glob(
    "/dccstor/lm4si/rudra/models/sarvam-2b-v0.5_instructlab_combined/hf_step_*"
)
combined_models = combined_models + glob(
    "/dccstor/lm4si/rudra/models/Llama-3.1-8B_instructlab_combined/hf_step_*"
)
combined_models = combined_models + glob(
    "/dccstor/lm4si/rudra/models/Llama-3.1-8B_instructlab_ashish/hf_step*"
)

questions = [
    "घर्षण बलों का पिण्डों की गति पर क्या प्रभाव पड़ता है?",
    "उल्का पात के विषय में प्रचलित विविध मान्यताएँ किन प्रकार के वृतांतों से जुड़ी हुई हैं?\na) पुराकथाओं, प्राचीन पौराणिक तथा अर्धएतिहासिक वृतांतों और लोककथाओं\nb) वैज्ञानिक तथ्यों के बावजूद\nc) भविष्यसूचक विश्वासपरंपरा से आबद्ध फलित ज्योतिष\nd) आकाश की इस विलक्षण घटना के फलाफल का विधान",
    "क्या चुम्बकत्व को एक प्रकार का प्राकृतिक बल माना जाता है जो वस्तुओं को एक दूसरे की ओर आकर्षित करता है?",
    "अक्रिस्टलीय ठोस क्या होते हैं और उनकी विशेषताएं क्या हैं?",
    "चिकित्सा भौतिकी में शोध का क्या महत्व है?",
    "क्रांतिक ताप से कम ताप पर गैस को द्रव में बदलने के लिए क्या करना होता है?\na) दाब बढ़ाना\nb) ताप बढ़ाना\nc) दाब घटाना\nd) केवल दाब बढ़ाने से",
    "वायुमंडलीय दाब पर हीलियम का क्वथनांक क्या है?\na) ०°K\nb) ४°K\nc) १०°K\nd) २०°K",
    "एकसमान और असमान गति की परिभाषा दें। दोनों के उदाहरण दें।",
    "चाल और वेग में अंतर बताइए।",
    "न्यूटन का पहला गति नियम क्या है? एक उदाहरण देकर इसे समझाएँ।",
    "जड़त्व की परिभाषा दें और इसके प्रकारों को समझाइए।",
    "न्यूटन के तृतीय नियम की व्याख्या एक उदाहरण के साथ करें।",
    "सार्वभौमिक गुरुत्वाकर्षण का नियम क्या है? इसके महत्व को समझाइए।",
    "ऊँचाई के साथ गुरुत्वाकर्षण के त्वरण में क्या परिवर्तन होता है?",
    "द्रव्यमान और भार में क्या अंतर है? एक उदाहरण के साथ समझाएँ।",
    "मुक्त पतन की परिभाषा दें। यह गुरुत्वाकर्षण के त्वरण से कैसे संबंधित है?",
    "कार्य की परिभाषा दें। क्या तब कार्य होता है जब एक आदमी अपने हाथ में सूटकेस लेकर चलता है? समझाएँ।",
    "ऊर्जा संरक्षण के नियम को एक उदाहरण के साथ समझाएँ।",
    "शक्ति और ऊर्जा में अंतर करें। शक्ति का सूत्र लिखें।",
]

gold_response = [
    "घर्षण बल पिण्डों की गति का विरोध करते हैं। जब कोई पिण्ड गतिशील होता है, तो घर्षण बल उसकी गति को कम करने की कोशिश करता है और अन्ततः उसे रोक देता है। इसलिए, पिण्डों को गतिशील बनाए रखने हेतु घर्षण बलों को निष्फल करने हेतु बाह्य साधनों द्वारा बल लगाना आवश्यक होता है।",
    "a) पुराकथाओं, प्राचीन पौराणिक तथा अर्धएतिहासिक वृतांतों और लोककथाओं\n\n",
    "हाँ",
    "अक्रिस्टलीय ठोस वे ठोस होते हैं जिनमें कणों की व्यवस्था निश्चित और नियमित नहीं होती है। इनमें लघु परास व्यवस्था पाई जाती है और इनका गलनांक अनिश्चित होता है। ये एक निश्चित ताप पर द्रव अवस्था में नहीं बदलते और ताप बढ़ाने पर धीरे-धीरे पिघलने लगते हैं।\n\n",
    "चिकित्सा भौतिकी में शोध का महत्व यह है कि यह नई तकनीकों और तरीकों को विकसित करने में मदद करता है जो स्वास्थ्य देखभाल में सुधार करने में उपयोग किए जा सकते हैं। इससे रोगों का निदान और उपचार बेहतर ढंग से किया जा सकता है और मरीजों को बेहतर देखभाल प्रदान की जा सकती है।\n\n",
    "d) केवल दाब बढ़ाने से \n\n",
    "b) ४°K\n\n",
]

answers_by_model = {}
answers_by_model["question"] = []
for each_question in questions:
    answers_by_model["question"].append(each_question)

answers_by_model["gold_response"] = []
for each_response in gold_response:
    answers_by_model["gold_response"].append(each_response)

for index in range(
    len(answers_by_model["gold_response"]), len(answers_by_model["question"])
):
    answers_by_model["gold_response"].append("")

for model_name in tqdm(models, desc="For each model"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.generation_config.pad_token_id = tokenizer.eos_token_id

        answers_by_model[model_name] = []

        for each_question in questions:
            if "_instructlab" in model_name:
                formatted_question = format_text(each_question)
            else:
                formatted_question = each_question
            tokenized_chat = tokenizer(formatted_question, return_tensors="pt")
            for each_key in tokenized_chat:
                tokenized_chat[each_key] = tokenized_chat[each_key].to(device)

            outputs = model.generate(
                **tokenized_chat,
                max_new_tokens=200,
                pad_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0])
            response = response.split(formatted_question)[-1]

            answers_by_model[model_name].append(response)
    except:
        pass

for model_name in tqdm(combined_models, desc="For each combined model"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        answers_by_model[model_name] = []

        for each_question in questions:
            formatted_question = format_text_prompt(each_question)
            tokenized_chat = tokenizer(formatted_question, return_tensors="pt")
            for each_key in tokenized_chat:
                tokenized_chat[each_key] = tokenized_chat[each_key].to(device)

            outputs = model.generate(
                **tokenized_chat,
                max_new_tokens=200,
                pad_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0])
            response = response.split(formatted_question)[-1]

            answers_by_model[model_name].append(response)
    except:
        pass

df = pd.DataFrame(answers_by_model)

with pd.ExcelWriter("InstructLab_Generations.xlsx") as writer:
    df.to_excel(writer, "Results", index=False)
