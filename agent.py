from lineage_tool import lineage
from langchain.tools import Tool, tool

import os
import json
from typing import Dict, Any, Optional, List
from enum import Enum
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")

with open("Event_Type_new_rough_lineage.csv", "r", encoding="utf-8") as file:
    csv_string = file.read()
lineage_doc = csv_string

@tool
def lineage_generation():
    """Generates the lineage document"""
    global lineage_doc
    lineage_doc = lineage()
    return "Lineage has been successfully generated and stored in the system."

llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-05-01-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

@tool 
def generate_pyspark_code():
    """Generates the pyspark code for the lineage document"""
    global llm
    global lineage_doc
    # Step 2: Create a prompt template
    prompt = ChatPromptTemplate.from_template("""You are an expert ETL developer with deep knowledge of both Informatica PowerCenter and Apache PySpark. Your task is to convert the provided Informatica mapping lineage document into a functional, clean, and well-documented PySpark script.

Analyze the provided CSV data, which details the end-to-end lineage from source to target. Each row represents a link between two components in an Informatica mapping.

**Follow these instructions precisely:**

1.  **Overall Structure:**
    * The final output must be a single PySpark script.
    * Start with the necessary imports (`pyspark.sql`, `pyspark.sql.functions as F`, etc.).
    * Define placeholder variables at the top of the script for Informatica parameters like `$$LANGUAGE_CODE`, `$$ETL_PROC_WID`, etc. This makes the script configurable.
    * Create a placeholder for reading the source data (`W_WRKFC_EVENT_TYPE_DS`) into a DataFrame named `source_df`.

2.  **Transformation Logic:**
    * Sequentially reconstruct the data flow based on the `FromInstance` and `ToInstance` columns. Create a new DataFrame for each major transformation step (e.g., `exp_validate_df`, `lkp_joined_df`, `final_df`).
    * **Expressions (`Expression`):** Translate the logic in the `Transformation` column into PySpark using `withColumn()` and functions from `pyspark.sql.functions`.
    * **Filters (`Filter`):** Implement these using the `.filter()` or `.where()` method.
    * **Lookups (`Lookup Procedure`):**
        * For each lookup, parse the `Lookup Sql Override` in the `Transformation` column.
        * Represent the lookup query as a new DataFrame (e.g., `lkp_eventtype_df = spark.sql("...")`).
        * Perform a **left join** from the main data flow DataFrame to the lookup DataFrame.
        * The join condition is based on the fields connecting to the lookup (e.g., `EVENT_REASON_CODE` going into `LKP_EventReason`).
    * **Update Strategy (`Update Strategy`):**
        * The logic `IIF(UPDATE_FLG = 'I' OR UPDATE_FLG = 'B', DD_INSERT, ...)` determines the action for each row.
        * Create a new column, for instance `operation_flag`, based on the `UPDATE_FLG` field. Map `DD_INSERT` to 'insert', `DD_UPDATE` to 'update', and `DD_REJECT` to 'reject'.
    * **Mapplets (`Mapplet`):** Treat mapplets as logical containers. Follow the lineage links into and out of the mapplet to implement its logic directly within the main script flow.

3.  **Function and Variable Translation:**
    * Translate Informatica functions to their PySpark equivalents:
        * `IIF(condition, true_val, false_val)` -> `F.when(condition, true_val).otherwise(false_val)`
        * `ISNULL(column)` -> `F.col(column).isNull()`
        * `SESSSTARTTIME` -> `F.current_timestamp()`
        * `LENGTH(str)` -> `F.length(F.col(str))`
    * Handle Informatica variables (`$$...`) as the configurable script variables defined at the beginning.

4.  **Final Output:**
    * The final DataFrame should select and name the columns as specified in the links to the `Target Definition` (`W_WRKFC_EVENT_TYPE_D`).
    * Include comments in the code that reference the Informatica transformation name (e.g., `# Applying logic from Exp_W_WRKFC_EVENT_TYPE_Transform`) for clarity and traceability.
    * Conclude with a placeholder showing how to write the final DataFrame, perhaps partitioning by the `operation_flag` or demonstrating a `MERGE` operation for a data lakehouse table (like Delta Lake).

Here is the Informatica lineage data:
```csv
{lineage}
```""")
    # Step 3: Bind LLM to the prompt
    chain = prompt | llm | StrOutputParser()
    # Step 4: Invoke the chain
    result = chain.invoke({"lineage": lineage_doc})
    with open("event_type_pyspark.txt","w") as file:
        file.write(result)
    print("successfully stored the pyspark code")


def create_agent():
    """
    Initialize the AI agent with Azure OpenAI and LangChain
    
    Args:
        azure_endpoint: Azure OpenAI endpoint URL
        api_key: Azure OpenAI API key
        api_version: API version (default: "2024-02-01")
        deployment_name: Model deployment name (default: "gpt-4")
    """
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("ERROR: AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables must be set.")
        return None
    
    # Initialize Azure OpenAI LLM
    global llm
    
    # Create tools
    tools = [lineage_generation,generate_pyspark_code]
    # Create agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI assistant specialized in converting Informatica XML files to PySpark code.

You can perform 2 tasks:
1. Generate the lineage document for the given XML using the lineage_generation tool.
2. Generate the pyspark code from the lineage document using generate_pyspark_code.


Available tools:
- lineage_generation: Generate lineage document from XML file  
"""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor

agent = create_agent()
# result = agent.invoke({"input": "Generate the lineage document"})
agent.invoke({"input": "The lineage is alreday generated and stored in the system. Generate the pyspark code for the lineage by invoking the generate_pyspark_code function"})


