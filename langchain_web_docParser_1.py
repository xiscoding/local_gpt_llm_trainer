#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:52:00 2024

@author: xdoestech
"""

import os
import requests
import re
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain import hub
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = "-"
TAVILY_API_KEY ="-"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

TEST_URL = "https://docs.angr.io/en/latest/"

GOAL = "Find url for the solution script for: Beginner reversing example: little_engine"

#Recursively load 20 urls
url = TEST_URL
loader = RecursiveUrlLoader(
    url=url, max_depth=3, extractor=lambda x: Soup(x, "html.parser").text
)
docs_RUL = loader.load()
documents_RUL = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs_RUL)
vector_RUL = FAISS.from_documents(documents_RUL, OpenAIEmbeddings())
retriever_RUL = vector_RUL.as_retriever()

retriever_tool = create_retriever_tool(
    retriever_RUL,
    "angr_search",
    """
    description: finds some relevant text.
    use: If you need to find information about angr use this tool!.
    """,
)
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
tools = [retriever_tool]
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "Find url for the solution script for: Beginner reversing example: little_engine"})