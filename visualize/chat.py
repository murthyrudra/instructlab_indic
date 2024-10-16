import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = []):
      super().__init__()
      self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
      for stop_id in self.stops:
        if stop_id in input_ids[0]:
            return True

      return False

st.title("Physics ChatBot")

# Options to select the model
model_option = st.selectbox(
    'Which model do you wish to chat?',
    ('Sarvam Base', 'Sarvam Physics Bot')
)

if 'previous_model' not in st.session_state:
    print("st.session_state is empty. Loading Sarvam Base model...")
    st.session_state.previous_model = "Sarvam Base"
    model_name = "sarvamai/sarvam-2b-v0.5"
    with st.spinner(f'Loading Model {model_name}...'):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        st.session_state.messages = []
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.is_base_model = True
        st.session_state.model_name = model_name

    with st.container():
        st.text(f"Finished Loading Model {st.session_state.model_name}")

    st.session_state.model.to('cuda')

if model_option != st.session_state.previous_model:
    print(f"Model changed {model_option}")
    if model_option == "Sarvam Base":
        model_name = "sarvamai/sarvam-2b-v0.5"
        with st.spinner('Loading Model...'):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            st.session_state.messages = []
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.is_base_model = True
            st.session_state.model_name = model_name
    else:
        model_name = "/dccstor/cssblr/rmurthyv/IBM/InstructLab/instructlab/output/instructlab/models/sarvam-2b-v05-trained/"
        with st.spinner('Loading Model...'):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            st.session_state.messages = []
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.is_base_model = False
            st.session_state.model_name = model_name

    st.session_state.model.to('cuda')

    with st.container():
        st.text(f"Finished Loading Model {st.session_state.model_name}")
    st.session_state.previous_model = model_option

def format_text(obj):
    return f"""\
<|system|>
I am, Red Hat\u00ae Instruct Model based on Granite 7B, an AI language model developed by Red Hat and IBM Research, based on the Granite-7b-base language model. My primary function is to be a chat assistant.
<|user|>
{obj['content']}
"""


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    stop_sequences = ["\n", "<|assistant|>", "<|user|>"]
    stop_words_ids = [st.session_state.tokenizer.encode(stop_word)[0] for stop_word in stop_sequences]
    st.session_state.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
# text = परमाणु प्रतिक्रिया क्या है?
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    print(st.session_state.is_base_model)
    print(st.session_state.messages)
    
    with st.spinner('Generating Response...'):
        if not st.session_state.is_base_model:
            message = format_text(st.session_state.messages[0])
        else:
            message = ""
            for each_message in st.session_state.messages:
                message = message + each_message["content"] + "\n"
        
        print(message)
        tokenized_chat = st.session_state.tokenizer(message, return_tensors="pt")

        for key in tokenized_chat:
            tokenized_chat[key] = tokenized_chat[key].to('cuda')

        if st.session_state.model_name == "sarvam-2b-v05-trained":
            outputs = st.session_state.model.generate(**tokenized_chat, max_new_tokens=200) 
        else:
            outputs = st.session_state.model.generate(**tokenized_chat, max_new_tokens=200) 
        response = st.session_state.tokenizer.decode(outputs[0])
        response = response.split(prompt)[-1]

        # if "<|user|>" in response:
        #     response = response.split("<|user|>")[0]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})