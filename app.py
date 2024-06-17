import os

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.llms import CTransformers

load_dotenv()

def getLLMResponse(form_input, email_sender, email_recipient, email_style):
    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")

    # Wrapper for Llama-2-7B-Chat, Running Llama 2 on CPU
    # https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
    # current_file_directory = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.abspath(os.path.join(current_file_directory,"models/llama-2-7b-chat.ggmlv3.q8_0.bin"))
    # llm = CTransformers(model=model_path,
    #                 model_type='llama',
    #                 config={'max_new_tokens': 256,
    #                         'temperature': 0.01})

    template = """
    Write a email with {style} style and includes topic :{email_topic}.\n\nSender: {sender}\nRecipient: {recipient}
    \n\nEmail Text:
    
    """

    prompt = PromptTemplate(
        input_variables=["style", "email_topic", "sender", "recipient"],
        template=template,
    )

    response = llm.invoke(
        prompt.format(
            email_topic=form_input,
            sender=email_sender,
            recipient=email_recipient,
            style=email_style,
        )
    )
    print(response)
    content = response.content
    return content


st.set_page_config(
    page_title="Generate Emails",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.header("Generate Emails 📧")

form_input = st.text_area("Enter the email topic", height=275)

# Creating columns for the UI - To receive inputs from user
col1, col2, col3 = st.columns([10, 10, 5])
with col1:
    email_sender = st.text_input("Sender Name")
with col2:
    email_recipient = st.text_input("Recipient Name")
with col3:
    email_style = st.selectbox(
        "Writing Style", ("Formal", "Appreciating", "Not Satisfied", "Neutral"), index=0
    )


submit = st.button("Generate")

# When 'Generate' button is clicked, execute the below code
if submit:
    st.write(getLLMResponse(form_input, email_sender, email_recipient, email_style))
