import os
import re
import logging
from datetime import datetime, UTC
from typing import Optional, Type

import requests
import gradio as gr
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

# Langchain imports
from langchain_core.runnables.utils import Output
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chains import LLMMathChain
from langchain.globals import set_verbose
from langchain.tools import StructuredTool
from langchain_community.tools import ShellTool, YouTubeSearchTool
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import (
    ArxivAPIWrapper,
    OpenWeatherMapAPIWrapper,
    WikipediaAPIWrapper
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import BaseTool, ToolException
from langchain_experimental.utilities import PythonREPL
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.language_models import BaseLLM

# Local imports
from helpers.vlm import get_image_description, get_image_description_url
from helpers.pdf_reader import PdfReader
from helpers.llm_provider import LLMProvider
from helpers.pdf_summarizer import PdfSummarizer

load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

session_id = None

def count_vowel(input: str) -> int:
    """Count the number of vowels in a string"""
    vowels = "aeiou"
    count = 0
    for char in input.lower():
        if char in vowels:
            count += 1
    return count

class LetterInWordCounterInput(BaseModel):
    input_string: str = Field(..., description="Input in the format 'word, letter'(without any quotes)")

class LetterInWordCounter(BaseTool):
    name: str = "LetterInWordCounter"
    description: str = "Counts occurrences of a letter in a word. Input should be in the format 'word, letter'(without any quotes)"
    args_schema: Type[BaseModel] = LetterInWordCounterInput
    return_direct: bool = True

    def _run(self, input_string: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> int:
        if input_string.startswith("'") and input_string.endswith("'"):
            input_string = input_string[1:-1]
            logger.info(f'after trimming:{input_string}')
        try:
            word, letter = input_string.lower().split(',')
            word = word.strip()
            letter = letter.strip()
            occ = word.count(letter)
            logger.info(f'input_string:{input_string}, occ:{occ}')
            return occ
        except ValueError as e:
            logger.exception(e)
            raise ToolException(f"Invalid input format {input_string}. Please provide input as 'word, letter'.")

    async def _arun(self, input_string: str) -> int:
        return self._run(input_string)

def recall_memory(last_n_messages: str) -> str:
    """Fetch Previous Conversations, to get all previous conversions input 0"""
    global session_id
    if "\n" in last_n_messages:
        last_n_messages = last_n_messages.split("\n")[0]
    logger.info(f'after trimming:{last_n_messages}')
    if last_n_messages.startswith("'") and last_n_messages.endswith("'"):
        last_n_messages = last_n_messages[1:-1]
    logger.info(f'after trimming:{last_n_messages}')
    if not last_n_messages.isnumeric():
        raise ToolException("Parameter 'last_n_messages' must be an integer")
    n = int(last_n_messages)
    n = 2*n
    history = get_message_history(session_id)
    if n == 0:
        n = len(history.messages)
    ordered_history = []
    for idx, message in enumerate(history.messages[::-1]):
        if idx >= n:
            break
        if type(message) == HumanMessage:
            ordered_history.append(f"Question: {message.content}\n")
        elif type(message) == AIMessage:
            ordered_history.append(f"Answer: {message.content}\n")
        else:
            message_type = type(message).__name__[:-7]
            ordered_history.append(f"{message_type}: {message.content}\n")
    history_string = ""
    for hist in ordered_history[::-1]:
        history_string += hist
    return history_string

def get_txt_file_content(file_path: str) -> str:
    """Reads the content of a file"""
    file_path = file_path.strip()
    file_path = os.path.normpath(file_path)
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise ToolException(f"File '{file_path}' not found.")
    except Exception as e:
        raise ToolException(f"Error reading file '{file_path}': {str(e)}")

def set_txt_file_content(input: str) -> str:
    file_path = input.split('|')[0].strip()
    file_content = input[len(file_path) + 1:]
    """Overwrites the content of a file. Syntax: file_path|file_content"""
    file_path = os.path.normpath(file_path)
    try:
        with open(file_path, 'w') as file:
            file.write(file_content)
        return "File updated."
    except FileNotFoundError:
        raise ToolException(f"File '{file_path}' not found.")
    except Exception as e:
        raise ToolException(f"Error reading file '{file_path}': {str(e)}")

def append_txt_file_content(input: str) -> str:
    file_path = input.split('|')[0].strip()
    file_content = input[len(file_path) + 1:]
    """Appends the content of a file. Syntax: file_path|file_content"""
    file_path = os.path.normpath(file_path)
    try:
        with open(file_path, 'a') as file:
            file.write(file_content)
        return "File updated."
    except FileNotFoundError:
        raise ToolException(f"File '{file_path}' not found.")
    except Exception as e:
        raise ToolException(f"Error reading file '{file_path}': {str(e)}")

def count_letter(input: str) -> int:
    """Count the number of letters in a string"""
    return len(re.findall(r'[a-zA-Z]', input))

def get_message_history(session_id: str) -> MongoDBChatMessageHistory:
    history = MongoDBChatMessageHistory(
        connection_string=os.getenv('MONGODB_URL'),
        session_id=session_id
    )
    
    messages = history.messages
    if len(messages) > 10:
        # Keep recent messages
        recent_messages = messages[-6:]  # Last 3 Q&A pairs
        # Summarize older messages using your LLM
        older_messages = messages[:-6]
        summary_prompt = "Summarize the following conversation keeping key points and context:\n"
        for msg in older_messages:
            summary_prompt += f"{msg.type}: {msg.content}\n"
        summary = llm.invoke(summary_prompt)
        # Clear history
        history.clear()
        # Add summary as system message
        history.add_message(SystemMessage(content=f"Previous conversation summary: {summary}"))
        # Add recent messages
        for msg in recent_messages:
            history.add_message(msg)
    return history

def notepad(command: str) -> str:
    """
    A simple notepad that can be read, write, or clear.

    Parameters:
    command (str): A string representing the command to be executed. The command can be one of the following:
        - ":READ:": Reads the content of 'notepad.txt' and returns it.
        - ":CLEAR:": Clears the content of 'notepad.txt'.
        - ":APPEND:<content>": Appends the given content to the end of 'notepad.txt'.
    """
    command = command.strip()
    if command == ":READ:":
        with open('helpers/notepad.txt', 'r') as file:
            content = file.read()
        return content
    elif command == ":CLEAR:":
        open('helpers/notepad.txt', 'w').close()
        return "Notepad cleared."
    elif command.startswith(":APPEND:"):
        content = command[8:].strip()
        with open('helpers/notepad.txt', 'a') as file:
            file.write(content + '\n')
        return "Notepad updated."
    else:
        return "Invalid command. Use :READ:, :CLEAR:, or :APPEND:<content>."

def web_browser(url: str) -> str:
    """Fetches the content of a webpage and returns a summary."""
    url = url.strip()
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "No title found"
        
        # Extract main content
        main_content = soup.find('body').get_text()
        
        # Summarize content
        summary = main_content[:10000] + "..." if len(main_content) > 10000 else main_content
        
        # Extract hyperlinks
        hyperlinks = [a.get('href') for a in soup.find_all('a', href=True)]

        result = f"Title: {title}\n\nSummary:\n{summary}\n\nHyperlinks:\n"
        result += "\n".join(hyperlinks)

        return result
    except Exception as e:
        raise ToolException(f"Error fetching webpage: {str(e)}")

def get_agent_prompt() -> PromptTemplate:
    template = '''
You are an intelligent assistant that uses step-by-step reasoning to solve problems.

Chat History:
{chat_history}

You have access to the following tools:

{tools}

Each tool is referenced by its name, which you should use when deciding to take an action. The tools you can use are: [{tool_names}]

When providing your answer, please follow this format:

Question: {input}
Thought: Consider the problem and think about how to solve it step by step.
Action: Decide if you need to use a tool to proceed. If so, specify the tool name from [{tool_names}].
Action Input: Provide input to the chosen tool if necessary.
Observation: Record the output from the tool.
... (Repeat Thought/Action/Action Input/Observation as needed)
Thought: Summarize your reasoning and arrive at the answer.
Final Answer: Provide the final answer in markdown format.

Remember to be thorough in your reasoning and ensure each step logically follows from the previous one.

Begin!

{agent_scratchpad}
'''
    return PromptTemplate.from_template(template)

def get_tools(llm: BaseLLM) -> list[Tool]:
    tavily_search = TavilySearchResults()
    wikipedia = WikipediaAPIWrapper()
    pubmed = PubmedQueryRun()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)
    arxiv = ArxivAPIWrapper()
    python_repl = PythonREPL()
    weather = OpenWeatherMapAPIWrapper()
    letter_in_wc = LetterInWordCounter()
    pdf_summarizer = PdfSummarizer(llm=llm)
    pdf_reader = PdfReader()
    shell_tool = ShellTool()
    youtube = YouTubeSearchTool()
    return [
        Tool.from_function(
            name="PythonREPL",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command.",
            handle_tool_error=True,
            func=python_repl.run,
        ),
        Tool.from_function(
            name="Terminal",
            description="A terminal in this computer. Use this to execute shell commands. Input should be a valid shell command.",
            handle_tool_error=True,
            func=shell_tool.run,
        ),
        Tool.from_function(
            name = "Weather",
            func=weather.run,
            handle_tool_error=True,
            description="Get the current weather information for a specified location"
        ),
        Tool.from_function(
            name = "Search",
            func=tavily_search.run,
            handle_tool_error=True,
            description="useful for answering questions about current events or general web search"
        ),
        Tool.from_function(
            name="Calculator",
            func=llm_math_chain.run,
            handle_tool_error=True,
            description="useful for when you need to answer math questions"
        ),
        Tool.from_function(
            name="Wikipedia",
            func=wikipedia.run,
            handle_tool_error=True,
            description="useful for encyclopedic general knowledge"
        ),
        Tool.from_function(
            name="YouTube",
            func=youtube.run,
            handle_tool_error=True,
            description="Searches youtube and provides related video links"
        ),
        Tool.from_function(
            name="Arxiv",
            func=arxiv.run,
            handle_tool_error=True,
            description="useful for searching scientific articles on arxiv.org"
        ),
        Tool.from_function(
            name="DateTimeNow",
            func=lambda x: datetime.now(UTC).strftime("%B %d %Y - %H:%M:%S") + " UTC",
            description="Returns the current date and time",
        ),
        Tool.from_function(
            name="LetterInWordCounter",
            func=letter_in_wc.run,
            handle_tool_error=True,
            description="Counts occurrences of a letter in a word. Input: 'word, letter'"
        ),
        StructuredTool.from_function(
            name="WebpageBrowser",
            func=web_browser,
            handle_tool_error=True,
            description="Fetches the content of a webpage and returns a summary"
        ),
        StructuredTool.from_function(
            name="PubMed",
            func=pubmed.run,
            handle_tool_error=True,
            description="Queries medical information from PubMed"
        ),
        StructuredTool.from_function(
            name="RecallMemory",
            func=recall_memory,
            handle_tool_error=True,
            description="Fetches n previous conversations (0 returns all)"
        ),
        StructuredTool.from_function(
            name="VowelCounter",
            func=count_vowel,
            handle_tool_error=True,
            description="Counts the number of vowels in a string"
        ),
        StructuredTool.from_function(
            name="TotalLetterCounter",
            func=count_letter,
            handle_tool_error=True,
            description="Counts the number of letters in a string",
        ),
        StructuredTool.from_function(
            name="PdfSummarizer",
            func=pdf_summarizer.summarize_pdf_with_image,
            handle_tool_error=True,
            description="Summarizes a PDF file; returns markdown"
        ),
        StructuredTool.from_function(
            name="ImgDesc",
            func=get_image_description,
            handle_tool_error=True,
            description="Provides a detailed visual description of an image"
        ),
        StructuredTool.from_function(
            name="ImgDescOnline",
            func=get_image_description_url,
            handle_tool_error=True,
            description="Provides a detailed visual description of an online image"
        ),
        StructuredTool.from_function(
            name="TxtFileReader",
            func=get_txt_file_content,
            handle_tool_error=True,
            description="Reads the content of a text file"
        ),
        StructuredTool.from_function(
            name="TxtFileWriter",
            func=set_txt_file_content,
            handle_tool_error=True,
            description="Overwrites the content of a text file. Syntax: file_path|file_content"
        ),
        StructuredTool.from_function(
            name="TxtFileAppender",
            func=append_txt_file_content,
            handle_tool_error=True,
            description="Appends content to a text file. Syntax: file_path|file_content"
        ),
        StructuredTool.from_function(
            name="PdfReader",
            func=pdf_reader.get_pdf_content,
            handle_tool_error=True,
            description="Reads PDF content; returns text"
        ),
        StructuredTool.from_function(
            name="PdfReaderOnline",
            func=pdf_reader.get_pdf_content_from_web,
            handle_tool_error=True,
            description="Reads PDF content from URL; returns text"
        ),
        StructuredTool.from_function(
            name="Notepad",
            func=notepad,
            handle_tool_error=True,
            description="A simple notepad: :READ:, :CLEAR:, or :APPEND:<content>"
        )
    ]


set_verbose(True)
llm = LLMProvider(temperature=0.0).get_llm()
tools = get_tools(llm)
agent = create_react_agent(llm=llm, tools=tools, prompt=get_agent_prompt())
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax", 
    return_intermediate_steps=True,
    verbose=True)
agent_with_memory = RunnableWithMessageHistory(
    agent_executor, 
    get_message_history,
    input_messages_key="input", 
    history_messages_key="chat_history", 
    output_messages_key="output"
)

def get_agent_output(input: str, file_path: str, session_id: str, error_count: int = 0) -> Output:
    error_count += 1
    try:
        output = agent_with_memory.invoke({
            "input": f"{input}, {f'file_path:{file_path}' if file_path is not None else ''}",
            "chat_history": get_message_history(session_id),
        }, config = {"configurable": {"session_id": session_id}}, handle_parsing_errors=True, handle_tool_errors=True)
    except ValueError as e:
        if str(e).startswith("Could not parse LLM output: `"):
            output = {"output": str(e).removeprefix("Could not parse LLM output: `").removesuffix("`")}
        elif error_count < 4:
            output = get_agent_output(input, file_path, session_id, error_count)
        else:
            raise
    logger.info(output)
    return output

def agent_io(input: str, file_path: str) -> str:
    logger.info(f"User input: {input}, File path: {file_path}")
    global session_id
    session_id = "custom-session-id"
    output = get_agent_output(input, file_path, session_id)
    return output["output"]

def main():
    inputs = []
    inputs.append(gr.Textbox(label="Ask anything you wanna explore!"))
    inputs.append(gr.File(label="Select a File")) 
    research_output = gr.Markdown(label="Response", height=400)
    
    interface = gr.Interface(
        fn=agent_io,
        inputs=inputs,
        outputs=research_output,
        title="GenericAI",
        flagging_mode="never"
    )
    interface.launch(server_name='0.0.0.0', server_port=8080, share=False)


if __name__ == "__main__":
    main()

