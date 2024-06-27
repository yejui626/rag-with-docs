import langchain
from langchain.agents import AgentType, Tool, initialize_agent
import streamlit as st
from chat import Chat
from structured_chat import StructuredChat


def initialize_agent(llm, db):
    unstructured_chat = Chat(db)
    #structured_chat = StructuredChat(sav_dir="uploaded_files", batch_token=st.session_state.batch_token)
    tools = [
        Tool(
            name="Resume Evaluation Results DataFrame QA System",
            func=unstructured_chat.chat,
            description="useful for when you need to answer questions about the information of the candidates or evaluation results dataframe. Input should be a fully formed question.",
        ),
        Tool(
            name="Resume Parser Tool QA System",
            func=unstructured_chat.chat,
            description="useful for when you need to answer questions about the resume parser tool itself. Input should be a fully formed question.",
        ),
    ]
    agent = langchain.agents.initialize.initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    
    return agent