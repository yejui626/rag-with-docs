import pandas as pd
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser

# Initialize Azure LLM
llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo-16k",
    azure_endpoint=st.secrets['OPENAI_API_ENDPOINT'],
    openai_api_type="azure",
    openai_api_version="2023-07-01-preview"
)

# Streamlit UI for file upload
st.title("CSV File Uploader")
uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])

dataframes = {}
if uploaded_files:
    for uploaded_file in uploaded_files:
        df_name = uploaded_file.name.split('.')[0]
        dataframes[df_name] = pd.read_csv(uploaded_file)
        st.write(f"DataFrame: {df_name}")
        st.dataframe(dataframes[df_name].head())

# Convert the dataframes to markdown for the prompt
df_template = """```python
{df_name}.head().to_markdown()
>>> {df_head}
```"""
df_context = "\n\n".join(
    df_template.format(df_head=_df.head().to_markdown(), df_name=df_name)
    for df_name, _df in dataframes.items()
)

# Define tool
tool = PythonAstREPLTool(locals=dataframes)
llm_with_tool = llm.bind_tools(tools=[tool], tool_choice=tool.name)

# Define system prompt
system_prompt = f"""You have access to multiple pandas dataframes. 
Here is a sample of rows from each dataframe and the python code that was used to generate the sample:

{df_context}

Given a user question about the dataframes, write the Python code to answer it. 
Don't assume you have access to any libraries other than built-in Python ones and pandas. 
Make sure to refer only to the variables mentioned above."""

# Define chat prompt template
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])

# Define parser
parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)

# Define function to get chat history
def _get_chat_history(x: dict) -> list:
    ai_msg = x["ai_msg"]
    tool_call_id = x["ai_msg"].additional_kwargs["tool_calls"][0]["id"]
    tool_msg = ToolMessage(tool_call_id=tool_call_id, content=str(x["tool_output"]))
    return [ai_msg, tool_msg]

# Define chain
chain = (
    RunnablePassthrough.assign(ai_msg=prompt | llm_with_tool)
    .assign(tool_output=itemgetter("ai_msg") | parser | tool)
    .assign(chat_history=_get_chat_history)
    .assign(response=prompt | llm | StrOutputParser())
    .pick(["tool_output", "response"])
)

# Chatbot interaction
st.title("Chatbot Interaction")
user_question = st.text_input("Ask a question about the dataframes:")

if user_question:
    result = chain.invoke({"question": user_question})
    st.write("Python Code:")
    st.code(result['tool_output'], language='python')
    st.write("Chatbot Response:")
    st.write(result['response'])

