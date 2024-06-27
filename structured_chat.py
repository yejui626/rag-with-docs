from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import os
from langchain_core.prompts import format_document
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st

class StructuredChat:
    def __init__(self, sav_dir:str, batch_token:str):
        self.sav_dir = sav_dir
        self.batch_token = batch_token
        self.llm = AzureChatOpenAI(
            deployment_name = "gpt-35-turbo-16k",
            azure_endpoint=st.secrets['OPENAI_API_ENDPOINT'],
            openai_api_type="azure",
            openai_api_version = "2023-07-01-preview"
        )
        self.memory = ChatMessageHistory(session_id="test-session")
        self.df = pd.read_excel(
            os.path.join(self.sav_dir, 'post_criteria_evaluation.xlsx')
        )
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            max_execution_time=10,
            max_iterations=2,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        self.agent_with_chat_history = RunnableWithMessageHistory(
            self.agent,
            # This is needed because in most real world scenarios, a session id is needed
            # It isn't really used here because we are using a simple in memory ChatMessageHistory
            lambda session_id: self.memory,
            input_messages_key="input",
            history_messages_key="message_history",
        )

    def chat(self, user_question):
        try:
            result = self.agent_with_chat_history.invoke(
                {"input": user_question},
                config={"configurable": {"session_id": self.batch_token}},
            )
            return result['output']
        except Exception as e:
            print(e)
            return 'Sorry, I couldnt answer this question for you :('
