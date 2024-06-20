import streamlit as st
import time
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import format_document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def pdf_loader(db,file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=100, add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)

    db.add_documents(documents=all_splits)
    
    


# Define the on_change function
def on_files_uploaded():
    st.session_state.uploaded_files = st.session_state.file_uploader
    st.session_state.file_descriptions = [""] * len(st.session_state.uploaded_files)

    

    
    


