from utils import _combine_documents
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
import streamlit as st

class Chat:
    def __init__(self, db):
        self.db = db

    def chat(self, prompt):
        memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
        )

        llm = AzureChatOpenAI(
            deployment_name = "gpt-35-turbo-16k",
            azure_endpoint=st.secrets['OPENAI_API_ENDPOINT'],
            openai_api_type="azure",
            openai_api_version = "2023-07-01-preview"
        )

        with st.spinner(text="Analyzing the documents... This should take 1-2 minutes."):
            retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k":2})
            _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:"""
            CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

            template = """Answer the question based only on the following context:
            {context}

            Question: {question}
            """
            ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

            standalone_question = {
                "standalone_question": {
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: get_buffer_string(x["chat_history"]),
                }
                | CONDENSE_QUESTION_PROMPT
                | llm
                | StrOutputParser(),
            }

            # Now we retrieve the documents
            retrieved_documents = {
                "docs": itemgetter("standalone_question") | retriever,
                "question": lambda x: x["standalone_question"],
            }
            # Now we construct the inputs for the final prompt
            final_inputs = {
                "context": lambda x: _combine_documents(x["docs"]),
                "question": itemgetter("question"),
            }
            # And finally, we do the part that returns the answers
            answer = {
                "answer": final_inputs | ANSWER_PROMPT | llm,
                "docs": itemgetter("docs"),
            }
            # And now we put it all together!
            final_chain = loaded_memory | standalone_question | retrieved_documents | answer
            inputs = {"question": prompt}
            result = final_chain.invoke(inputs)
            memory.save_context(inputs, {"answer": result["answer"].content})
            return result["answer"].content