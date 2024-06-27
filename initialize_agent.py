from langchain.agents import AgentType, Tool, initialize_agent
import streamlit as st


def initialize_agent(llm,tools):
    tools = [
        Tool(
            name="Resume Evaluation Results DataFrame QA System",
            func=post_evaluation_chat.run,
            description="useful for when you need to answer questions about the information of the candidates or evaluation results dataframe. Input should be a fully formed question.",
        ),
        Tool(
            name="Resume Parser Tool QA System",
            func=default_chat.run,
            description="useful for when you need to answer questions about the resume parser tool itself. Input should be a fully formed question.",
        ),
    ]
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    
    return agent