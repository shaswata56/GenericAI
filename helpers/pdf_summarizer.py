from langchain_core.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain_ollama.llms import OllamaLLM
from io import StringIO
import pymupdf4llm
import pathlib
import os
import re

from helpers.vlm import get_image_description
from helpers.pdf_reader import PdfReader


class PdfSummarizer():

    llm: BaseLLM
    reader: PdfReader

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.reader = PdfReader()

    def summarize_pdf_with_image(self, pdf_file_path: str) -> str:
        print("Started")
        has_img = False
        pdf_file_path = pdf_file_path.strip()
        pdf_file_path = os.path.normpath(pdf_file_path)
        document = self.reader.get_pdf_content(pdf_file_path)
        if "<imgdesc>" in document:
            has_img = True
        prompt_template = f"""
        As a professional summarizer, create a concise and comprehensive summary of the provided text:
            * Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness.
            * Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.
            * Rely strictly on the provided text, without including external information or reference.
            * Format the summary in key-points form for easy understanding.
        The text is provided in markdown format between <text> and </text> tag.
        {'Description of images/figures in this document are provided in text between between <imgdesc> and </imgdesc> tag' if has_img else ''}
        Return response which covers the key points of the text.
        <text>{'{text}'}</text>
        """
        summarize_prompt = PromptTemplate(template=prompt_template, input_variables=['text'])
        chain = (summarize_prompt | self.llm)
        output = chain.invoke({"text": document})
        print(output)
        print("Done")
        if isinstance(self.llm, OllamaLLM):
            return output
        return output.content
