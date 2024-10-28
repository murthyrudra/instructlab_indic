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
"I am, Hindi Instruct Model based on Sarvam 2B, an AI language model developed by Sarvam and fine-tuned by IBM Research, based on the Llama-3.1 70B Instruct language model. My primary function is to be able to answer Physics based questions."
<|user|>
{obj}
<|assistant|>"""


models = [
    "meta-llama/Llama-3.1-8B",
]
combined_models = models + glob(
    "/dccstor/cssblr/rmurthyv/IBM/dolomite-engine/output/fine_tuned/Meta-Llama-3.1-8B_instructlab_InstructLabPhysics_v1/epoch_*"
)

combined_models = combined_models + glob(
    "/dccstor/cssblr/rmurthyv/IBM/dolomite-engine/output/fine_tuned/Meta-Llama-3.1-8B_fullft_instructlab_InstructLabPhysics_v1/epoch*"
)


questions = [
    "पृथ्वी की प्लेटों की सीमाओं पर भूकम्प आने की प्रवृत्ति क्यों होती है?",
    "रिक्टर पैमाने पर भूकम्प की माप क्या बताता है?",
    "दो विद्युत धारावाही चालकों के बीच चुंबकीय बल की प्रकृति क्या है, और इसकी खोज किसने की थी?",
    "विद्युत धारावाही चालकों पर चुंबकीय बल की प्रकृति क्या है, और यह किन कारकों से प्रभावित होता है?",
    "ऐम्पियर ने किस प्रकार के अध्ययन किए थे, और उनके अध्ययनों का क्या महत्व है?",
    "प्लास्टर ऑफ पेरिस का रासायनिक सूत्र क्या है, और इसका निर्माण कैसे होता है?",
    "अम्ल-क्षारक सूचक रंजक क्या होते हैं?",
    "हर्ट्ज ने किस प्रकार की तरंगों की खोज की?",
    "प्रकाश की प्रकृति क्या है?",
    "रोटी बनाने की पूरी प्रक्रिया में कौन-कौन से लोग सम्मिलित होते हैं?",
    "गैल्वेनोमीटर और ऐमीटर के बीच क्या मुख्य अंतर है, और इन दोनों को एक दूसरे में परिवर्तित करने के लिए किन संशोधनों की आवश्यकता होती है?",
    "दक्षिण हस्त अंगुष्ठ नियम क्या है और इसका क्या महत्व है?",
]

paraphrase_questions = [
    "भूकम्प पृथ्वी की प्लेटों की सीमाओं पर क्यों उत्पन्न होते हैं?",
    "रिक्टर पैमाने पर भूकम्प की माप से हमें क्या जानकारी मिलती है?",
    "दो विद्युत धारावाही चालकों के बीच चुंबकीय बल की विशेषताएँ क्या हैं, और इसकी खोज किसने की?",
    "चुंबकीय बल की विशेषताएँ विद्युत धारावाही चालकों पर क्या होती हैं, और ये किन तत्वों से प्रभावित होती हैं?",
    "ऐम्पियर ने किस तरह के शोध किए, और उनके कार्यों का क्या महत्व है?",
    "प्लास्टर ऑफ पेरिस का रासायनिक सूत्र क्या है, और इसे कैसे बनाया जाता है?",
    "अम्ल-क्षारक सूचक रंजक किसे कहते हैं?",
    "हर्ट्ज ने किस प्रकार की तरंगों की खोज की थी?",
    "प्रकाश की प्रकृति का क्या विवरण है?",
    "रोटी बनाने की प्रक्रिया में कौन-कौन से लोग शामिल होते हैं?",
    "गैल्वेनोमीटर और ऐमीटर में क्या प्रमुख अंतर है, और इन्हें एक-दूसरे में बदलने के लिए क्या परिवर्तन चाहिए?",
    "दक्षिण हस्त अंगुष्ठ नियम क्या है और इसका क्या महत्व है?",
]

gold_response = [
    "पृथ्वी की प्लेटें एक दूसरे के सापेक्ष गति करती हैं और जब ये प्लेटें एक दूसरे से टकराती हैं या अलग होती हैं, तो इससे ऊर्जा का संचय होता है जो भूकम्प के रूप में निकलती है।\n\n",
    "रिक्टर पैमाने पर भूकम्प की माप उसकी विनाशी ऊर्जा को दर्शाता है। 7 से अधिक माप वाले भूकम्प जीवन और सम्पत्ति की अपार क्षति कर सकते हैं।\n\n",
    "दो विद्युत धारावाही चालकों के बीच चुंबकीय बल की प्रकृति ऐम्पियर द्वारा खोजी गई थी। ऐम्पियर ने सन् 1820-25 की अवधि में इस चुंबकीय बल की प्रकृति, इसकी विद्युत धारा के परिमाण, चालक की आकृति तथा आमाप पर निर्भरता के साथ इन चालकों के बीच की दूरी पर निर्भरता का अध्ययन किया।\n\n",
    "विद्युत धारावाही चालकों पर चुंबकीय बल की प्रकृति बल आरोपित करने की होती है, जो उनकी विद्युत धारा, आकृति, दूरी, और उनकी स्थिति से प्रभावित होती है। यह बल लोरेंज बल सूत्र का अनुगमन करता है, जो विद्युत धारावाही चालकों के बीच की दूरी और उनकी विद्युत धारा के परिमाण पर निर्भर करता है।\n\n",
    "ऐम्पियर ने दो विद्युत धारावाही चालकों के बीच के चुंबकीय बल की प्रकृति, इसकी विद्युत धारा के परिमाण, चालक की आकृति, आमाप, और इन चालकों के बीच की दूरी पर निर्भरता का अध्ययन किया था। उनके अध्ययनों का महत्व यह है कि उन्होंने चुंबकीय बल की प्रकृति और इसके कारकों को समझने में हमारी सहायता की है।\n\n",
    "प्लास्टर ऑफ पेरिस का रासायनिक सूत्र CaSO .- HxO है। इसका निर्माण जिप्सम को 373 K पर गर्म करने से होता है। इस प्रक्रिया में, जिप्सम जल के अणुओं का त्याग कर कैल्सियम सल्फेट अर्धहाइड्रेट बनाता है।\n\n",
    "अम्ल-क्षारक सूचक रंजक वे रंजक या रंजकों के मिश्रण होते हैं जिनका उपयोग अम्ल और क्षारक की उपस्थिति को सूचित करने के लिए किया जाता है।\n\n",
    "हर्ट्ज ने रेडियो तरंगों की खोज की, जो विद्युत चुंबकीय तरंगों का एक प्रकार है।\n\n",
    "प्रकाश वास्तव में विद्युत चुंबकीय तरंगें हैं। यह बात जेम्स मैक्सवेल ने अपने नए नियमों में स्पष्ट की थी और यह अनुभव किया गया कि प्रकाश विद्युत चुंबकीय तरंगों का रूप है।\n\n",
    "रोटी बनाने की पूरी प्रक्रिया में किसान, मजदूर, व्यापारी, परिवहनकर्ता, आटा मिल के कर्मचारी, और घर के सदस्य जैसे कई लोग सम्मिलित होते हैं। इन सभी लोगों की मेहनत और योगदान से ही हमें रोटी मिलती है।\n\n",
    "गैल्वेनोमीटर और ऐमीटर के बीच मुख्य अंतर यह है कि गैल्वेनोमीटर वोल्टता को मापने के लिए उपयोग किया जाता है, जबकि ऐमीटर धारा को मापने के लिए उपयोग किया जाता है। गैल्वेनोमीटर को ऐमीटर में परिवर्तित करने के लिए, इसमें एक शंट प्रतिरोध जोड़ा जाता है, जिससे धारा सुग्राहिता में वृद्धि होती है। इसके विपरीत, ऐमीटर को गैल्वेनोमीटर में परिवर्तित करने के लिए, इसमें एक प्रतिरोध जोड़ा जाता है, जिससे वोल्टता सुग्राहिता में वृद्धि होती है। ",
    "दक्षिण हस्त अंगुष्ठ नियम एक नियम है जो विद्युत धारा की दिशा को ध्यान में रखते हुए चुंबकीय क्षेत्र की दिशा को निर्धारित करता है। इसका महत्व यह है कि यह हमें चुंबकीय क्षेत्र की दिशा को समझने में मदद करता है और इसका उपयोग विभिन्न विद्युत उपकरणों में किया जाता है।\n\n",
]

answers_by_model = {}
answers_by_model["is_paraphrase"] = []

# Add questions and gold response
answers_by_model["question"] = []
for each_question in questions:
    answers_by_model["question"].append(each_question)
    answers_by_model["is_paraphrase"].append(False)

for each_question in paraphrase_questions:
    answers_by_model["question"].append(each_question)
    answers_by_model["is_paraphrase"].append(True)

answers_by_model["gold_response"] = []
for each_response in gold_response:
    answers_by_model["gold_response"].append(each_response)

for each_response in gold_response:
    answers_by_model["gold_response"].append(each_response)


for index in range(
    len(answers_by_model["gold_response"]), len(answers_by_model["question"])
):
    answers_by_model["gold_response"].append("")

for model_name in tqdm(combined_models, desc="For each model"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    answers_by_model[model_name] = []

    # Training data questions
    for each_question in questions:
        tokenized_chat = tokenizer(each_question, return_tensors="pt")
        for each_key in tokenized_chat:
            tokenized_chat[each_key] = tokenized_chat[each_key].to(device)

        outputs = model.generate(
            **tokenized_chat,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0])
        response = response.split(each_question)[-1]

        answers_by_model[model_name].append(response)

    # Paraphrased questions
    for each_question in paraphrase_questions:
        tokenized_chat = tokenizer(each_question, return_tensors="pt")
        for each_key in tokenized_chat:
            tokenized_chat[each_key] = tokenized_chat[each_key].to(device)

        outputs = model.generate(
            **tokenized_chat,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0])
        response = response.split(each_question)[-1]

        answers_by_model[model_name].append(response)


df = pd.DataFrame(answers_by_model)

with pd.ExcelWriter("InstructLab_Generations.xlsx") as writer:
    df.to_excel(writer, "Results", index=False)
