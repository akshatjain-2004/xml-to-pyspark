from lineage_tool import lineage
from pyspark_generator import code_generator
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
    # Step 2: Create a prompt template
    code_generator()
    return "Successfullt generated the pyspark code and stored in system"

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


