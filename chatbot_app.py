import streamlit as st
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from streamlit_chat import message

st.set_page_config(layout="wide")

device = torch.device('cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)

persist_directory = "db"

import tempfile

import tempfile

@st.cache_resource
def data_ingestion(uploaded_file):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load data from the temporary file
    loader = PDFMinerLoader(temp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()
    db = None

    # Remove the temporary file
    os.unlink(temp_file_path)

# ... (rest of the code)


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction, qa):
    response = ''
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

import random

def display_conversation(history, file_name):
    for i in range(len(history["generated"])):
        random_number = random.randint(1, 1000000)
        user_key = f"{file_name}_{i}_user_{random_number}"
        generated_key = f"{file_name}_{i}_generated_{random_number}"
        message(history["past"][i], is_user=True, key=user_key)
        message(history["generated"][i], key=generated_key)





def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF </h2>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Choose multiple PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        file_histories = {}

        for uploaded_file in uploaded_files:
            if uploaded_file.name not in file_histories:
                file_histories[uploaded_file.name] = {"past": ["Hey there!"], "generated": ["I am ready to help you"]}

            file_details = {
                "Filename": uploaded_file.name,
                "File size": get_file_size(uploaded_file)
            }
            st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)

            display_conversation(file_histories[uploaded_file.name], uploaded_file.name)



            with st.spinner(f'Embeddings for {uploaded_file.name} are in process...'):
                file_histories[uploaded_file.name]["past"].append(uploaded_file.name)
                data_ingestion(uploaded_file)
            st.success(f'Embeddings for {uploaded_file.name} are created successfully!')

            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)
            user_input = st.text_input("", key=f"input_{uploaded_file.name}")

            if user_input:
                qa = qa_llm()
                answer = process_answer({'query': user_input}, qa)
                file_histories[uploaded_file.name]["past"].append(user_input)
                response = answer
                file_histories[uploaded_file.name]["generated"].append(response)

                display_conversation(file_histories[uploaded_file.name], uploaded_file.name)



if __name__ == "__main__":
    main()
