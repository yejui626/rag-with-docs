import pandas as pd
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from chat import Chat
from utils import pdf_loader,on_files_uploaded
from initialize_agent import initialize_agent


st.title("title bar : Chat with docs")
st.info("Info bar", icon="📃")
embedding_function = AzureOpenAIEmbeddings(
                    deployment = "ada002",
                    model="text-embedding-ada-002",
                    azure_endpoint=st.secrets['OPENAI_API_ENDPOINT'],
                    openai_api_version = "2023-07-01-preview"
                    )

llm = AzureChatOpenAI( # added bc llm needs to be passed to initialize_agent
            deployment_name="gpt-35-turbo-16k",
            azure_endpoint=st.secrets['OPENAI_API_ENDPOINT'],
            openai_api_type="azure",
            openai_api_version="2023-07-01-preview"
        )

def change_folder(folder):
    with st.spinner("Thinking..."):
        print("Folder name:",folder)
        db = Chroma(collection_name=folder,
                    persist_directory=f"directories/{folder}",
                    embedding_function=embedding_function
                    )
        st.session_state.chat_engine = initialize_agent(llm, db)
        
# Initialize session state variables
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'file_descriptions' not in st.session_state:
    st.session_state.file_descriptions = []
if 'batch_token' not in st.session_state: # to handle error in initialize_agent
    st.session_state.batch_token = "your_default_batch_token"


with st.sidebar:
    st.subheader('Document Chatbot!')
    with st.form("change_folder",clear_on_submit=False):
        # Define the directory path
        directory_path = "directories"

        # Get all directory names
        directory_names = [name for name in os.listdir(directory_path)]
        print("Folder names: ",directory_names)

        # Create the select box
        selected_directory = st.selectbox('Which directory do you want to chat with?', directory_names)

        # Display the selected directory
        if selected_directory != "Select a directory":
            st.write(f"You selected: {selected_directory}")

        change_folder_button = st.form_submit_button("Change Folder")
        if change_folder_button:
            change_folder(selected_directory)

    st.divider()
    uploaded_files = st.file_uploader(
        "Upload your PDF/XLSX documents here",
        type=['pdf', 'xlsx'],
        accept_multiple_files=True,
        on_change=on_files_uploaded,
        key='file_uploader'
    )
    with st.form("my_form",clear_on_submit=True):
        # Create a directory if it doesn't exist
        root_dir = "uploaded_files"
        os.makedirs(root_dir, exist_ok=True)
        # File uploader with on_change event
        folder_path = st.text_input(
            label=":rainbow[Folder Name]", 
            value="", 
            on_change=None, 
            placeholder="Insert your folder name here", 
            label_visibility="visible"
        )
        # Generate description inputs based on uploaded files
        if st.session_state.uploaded_files:
            st.subheader("Describe your documents")
            for i, file in enumerate(st.session_state.uploaded_files):
                description = st.text_input(
                    label=f"Description for {file.name}", 
                    value=st.session_state.file_descriptions[i], 
                    key=f"file_description_{i}"
                )
                st.session_state.file_descriptions[i] = description

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")

        if submitted:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Check if files are uploaded
                    if uploaded_files is not None:
                        os.makedirs(f"{root_dir}/{folder_path}", exist_ok=True)
                        db = Chroma(collection_name=folder_path,
                                    persist_directory=f"directories/{folder_path}", 
                                    embedding_function=embedding_function
                                    )
                        
                        # Save all file descriptions as .csv
                        descriptions_df = pd.DataFrame({
                            "file_name": [file.name for file in uploaded_files],
                            "description": st.session_state.file_descriptions
                        })
                        descriptions_df.to_csv(f"{root_dir}/{folder_path}/file_descriptions.csv", index=False)
                        st.success("File descriptions saved successfully!")
                        
                        for uploaded_file in uploaded_files:
                            bytes_data = uploaded_file.read()
                            file_name = uploaded_file.name
                            file_type = uploaded_file.type
                            print(file_type)
                            file_path = os.path.join(root_dir, f"{folder_path}/{file_name}")
                            
                            # Write the contents of the file to a new file
                            with open(file_path, "wb") as f:
                                f.write(bytes_data)
                            
                            if file_type == "application/pdf":
                                pdf_loader(db=db,file_path=file_path)                            
                                
                    else:
                        print("FAILED")

    st.divider()

    cols = st.columns(1)
    if cols[0].button('Refresh'):
        print("session state before clearing",st.session_state)
        st.session_state.clear()
        print("session state after clearing",st.session_state)

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": f"Select your directory!"}
    ]

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    db = Chroma(collection_name="default", embedding_function=embedding_function, persist_directory="directories/default")
    st.session_state.chat_engine = initialize_agent(llm, db)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) # Add response to message history

