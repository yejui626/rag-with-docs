import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from chat import Chat


st.title("title bar : Chat with docs")
st.info("Info bar", icon="📃")
embedding_function = AzureOpenAIEmbeddings(
                    deployment = "ada002",
                    model="text-embedding-ada-002",
                    azure_endpoint=st.secrets['OPENAI_API_ENDPOINT'],
                    openai_api_version = "2023-07-01-preview"
                    )


def change_folder(folder):
    with st.spinner("Thinking..."):
        chroma_directory= os.path.join(os.getcwd(),f"directories/{folder}") # Change to your own file directory
        print("Folder name:",folder)
        db = Chroma(collection_name=folder,
                    persist_directory=chroma_directory, 
                    embedding_function=embedding_function
                    )
        index = Chat(db)
        print(index)
        st.session_state.chat_engine = index

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
    with st.form("my_form",clear_on_submit=True):
        # Create a directory if it doesn't exist
        save_dir = "uploaded_files"
        os.makedirs(save_dir, exist_ok=True)
        uploaded_files = st.file_uploader(
            "Upload your PDF documents here",
            type=['pdf'],  #Only Allowing PDF for now
            accept_multiple_files=True
        )
        folder_path = st.text_input(
            label=":rainbow[Folder Name]", 
            value="", 
            on_change=None, 
            placeholder="Insert your folder name here", 
            label_visibility="visible"
        )
        # folder_description = st.text_input(
        #     label=":rainbow[Folder Description]", 
        #     value="", 
        #     on_change=None, 
        #     placeholder="Describe your documents", 
        #     label_visibility="visible"
        # )

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")

        if submitted:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Check if files are uploaded
                    if uploaded_files is not None:
                        chroma_directory=os.path.join(os.getcwd(),f"directories/{folder_path}") # Change to your own file directory
                        db = Chroma(collection_name=folder_path,
                                    persist_directory=chroma_directory, 
                                    embedding_function=embedding_function
                                    )
                        for uploaded_file in uploaded_files:
                            bytes_data = uploaded_file.read()
                            file_name = uploaded_file.name
                            file_path = os.path.join(save_dir, file_name)
                            
                            # Write the contents of the file to a new file
                            with open(file_path, "wb") as f:
                                f.write(bytes_data)
                            
                            loader = PyPDFLoader(file_path)
                            documents = loader.load()

                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=10000, chunk_overlap=100, add_start_index=True
                            )
                            all_splits = text_splitter.split_documents(documents)

                            db.add_documents(documents=all_splits)
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
    default_directory=os.path.join(os.getcwd(),"directories/default") # Change to your own file directory
    print(default_directory)
    db = Chroma(collection_name="default",persist_directory=default_directory, 
                embedding_function=embedding_function
                )
    index = Chat(db)
    st.session_state.chat_engine = index

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


