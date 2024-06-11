from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


class UnstructuredChat:
    def run(self, user_question):
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
            embeddings = OpenAIEmbeddings()
            new_db = FAISS.load_local(folder_path='vector_stores',index_name = "faiss_index_default", embeddings=embeddings,allow_dangerous_deserialization=True)

            retriever = new_db.as_retriever()
            template = """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. The provided context is labelled with **Question** and **Answer**. Return everything under **Answer**.
            If you don't know the answer, just say that you don't know.

            Question: {question} 

            Context: {context} 

            Answer:
            """

            prompt = ChatPromptTemplate.from_template(template)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)


            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            result = rag_chain.invoke(user_question)

            return result
        except Exception as e:
            print(e)
            return 'Sorry, I couldnt answer this question for you :('
