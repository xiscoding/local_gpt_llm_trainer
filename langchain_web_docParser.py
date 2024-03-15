#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 03:37:40 2024

@author: xdoestech
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 19:03:53 2024

@author: xdoestech
"""
import os
import requests
import re
from bs4 import BeautifulSoup
import nltk
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import ShellTool
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import json
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
import pandas as pd


OPENAI_API_KEY = "-"
TAVILY_API_KEY ="-"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# strings / constants
context_from_url_template = """
Considering the user's query "{clean_query}", analyze the following list of links:

**Links:**

{prompt_post_urls}

Each link has a title ("text") and a URL ("href"). Based on the titles and the user's query, rank the links in order of how likely they are to contain relevant information.

For each link, explain why you ranked it in that position, considering the following:

* How well do the words in the title ("text") match the user's query "{clean_query}"?
* Are there any keywords in the title that suggest the content might be relevant to the query?

"""

FULL_TEMPLATE = """{main}

{example}

{start}"""
EXAMPLE_TEMPLATE = """Here's an example of the output format you will follow:

**Example output format:**

* {{"Rank":"1", "text": "A Beginner's Guide to {{clean_query}}", "href": "legitimatelink.htm", "Explanation":"This title directly mentions the query and suggests introductory content"}} 
* {{"Rank":"2", "text": "Advanced Techniques in {{Related Field}}", "href": "lookslikealink.com", "Explanation":"This title might be relevant if {clean_query} is a subtopic of the related field"}}

"""
START_TEMPLATE = """**Note:** It is safe to assume the titles are descriptive of the linked webpages.
**Note:** Do not rank more than {link_num} links.
**Note**The href contents must be at least 2 tokens in length."""

TEST_URL = "https://docs.angr.io/en/latest/"

test_query_1 = "Constraint Solving"
test_query_2 = "Usage"
test_query_3 = "Debugging"
###############################################################################

#Functions
def convert_to_json(links_with_text):
    # Convert the list of dictionaries into a JSON formatted string
    json_output = json.dumps(links_with_text, indent=None) #indent
    return json_output

def clean_text(text):
    # Remove all characters that aren't letters or numerics
    return re.sub('[^a-zA-Z0-9]', '', text)

def list_all_links_with_text(url):
    # Initialize WebBaseLoader with the URL
    loader = WebBaseLoader(url)

    # Scrape the webpage content
    soup = loader.scrape()  # This returns BeautifulSoup object

    # Find all <a> tags to extract links and their cleaned text
    links_with_text = []
    for link in soup.find_all('a'):
        href = link.get('href')
        text = link.get_text().strip() if link.get_text().strip() else "NONE"
        # Clean the text and href to remove non-text characters
        cleaned_text = clean_text(text) if text and text != "NONE" else "NONE"
        # cleaned_href = clean_text(href)
        links_with_text.append({"text": cleaned_text, "href": href})

    return links_with_text

def get_urls(url):

  # Initialize WebBaseLoader with the URL
    loader = WebBaseLoader(url)

    # Scrape the webpage content
    soup = loader.scrape()  # This returns BeautifulSoup object

    # Find all <a> tags to extract links and their cleaned text
    links_with_text = []
    for link in soup.find_all('a'):
        href = link.get('href')
        text = link.get_text().strip() if link.get_text().strip() else "NONE"
        # Clean the text and href to remove non-text characters
        cleaned_text = clean_text(text) if text and text != "NONE" else "NONE"
        # cleaned_href = clean_text(href)
        links_with_text.append({"text": cleaned_text, "href": href})

    return convert_to_json(links_with_text)

def process_data(data_string):
    """
    Processes a string containing formatted data and converts it to a DataFrame.

    Args:
        data_string: A string containing the data with specific formatting.

    Returns:
        A pandas DataFrame with columns "Rank", "text", "href", and "Explanation".
    """

    lines = data_string.splitlines()
    processed_data = []

    for line in lines:
        if not line:
          continue
        try:
            # Extract JSON string by removing leading non-JSON characters
            json_str = line[line.index('{'):]
            item_data = json.loads(json_str)

                    # Extract information
            rank = item_data["Rank"]
            text = item_data["text"]
            href = item_data["href"]
            explanation = item_data["Explanation"]

            # Create a temporary dictionary to store processed data for this item
            processed_item = {
              "Rank": rank,
              "text": text,
              "href_short": href,  # Keep original href for reference
              "explanation": explanation
            }
        except Exception as e:
          # Log or print the error
          print(f"Error processing line: {e}")
          print(f"line: {line}")
        # Prepend base URL to href and create a new column
        item_data["href_valid"] = f"https://docs.angr.io/en/latest/{href}"

          # Add the processed item to the list
        processed_data.append(item_data)
    # Create and return a DataFrame from the processed data
    df = pd.DataFrame(processed_data)
    return df

def create_invoke_get_urls(q_str, a_str, num_str):
    invoke_prompt = {"clean_query":q_str, "prompt_post_urls":a_str,
                                   "link_num":num_str}
    return invoke_prompt
###############################################################################


#GET FIVE RELEVANT URL LINKS FROM WEBPAGE
all_links = list_all_links_with_text(TEST_URL)

#convert url links to json txt
clean_links = convert_to_json(all_links)

#Setup Prompts
full_prompt = PromptTemplate.from_template(FULL_TEMPLATE)
instruction = context_from_url_template
main_prompt = PromptTemplate.from_template(instruction)
example_prompt = PromptTemplate.from_template(EXAMPLE_TEMPLATE)
start_prompt = PromptTemplate.from_template(START_TEMPLATE)
input_prompts = [
    ("main", main_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt, pipeline_prompts=input_prompts
)

#Create Chain
output_parser = StrOutputParser()
# output_parser = SimpleJsonOutputParser()
#ChatOpenAI(): default="gpt-3.5-turbo", alias="model"
model = ChatOpenAI()
prompt = pipeline_prompt
chain = prompt | model | output_parser | process_data
#Invoke Chain 
#OUTPUT STRING
num_str="5"
df = chain.invoke(create_invoke_get_urls(test_query_1, clean_links, num_str))
###############################################################################

#SEARCH 
# df = process_data(next_five_links)
# Visit each link and generate a summary and list of links
# Generate a summary of the contents of each link
##### use lanchain 
##use list_all_links_with_text(url) function to get list of links
##add new collumns to df or add to existing link summary and link list columns
#### "link summary": contains the llm generated summary
#### "link list": contains the list of links
###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

