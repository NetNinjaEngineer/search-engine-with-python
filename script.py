import os
import sys
import re
from typing import Dict, List, Optional, Union, Any
import string
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import queue
import PyPDF2
import csv
import json
import openpyxl
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import whoosh
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, STORED, NUMERIC
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.qparser import OrGroup, AndGroup
from whoosh.qparser.plugins import FuzzyTermPlugin, WildcardPlugin, PhrasePlugin
from whoosh.scoring import BM25F

class DocumentProcessor:
    def __init__(self, index_dir = "index"):
        self.index_dir = index_dir
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')

        self.create_index_dir()

        # create the schema for the index
        self.schema = Schema(
            path=ID(stored=True),
            filetype=ID(stored=True),
            location=ID(stored=True),
            content=TEXT(stored=True),
            score=NUMERIC(stored=True, sortable=True)
        )

        # create index
        self.create_or_open_index()

    def process_text(self, text: str) -> str:
        """Process the text by removing punctuation, stop words, and lemmatizing."""
        #Remove punctuation
        if not text:
            return ""
        
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stop words
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word.lower() not in stop_words]

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # POS tagging
        tagged_tokens = pos_tag(tokens)

        # Filter out proper nouns
        tokens = [word for word, tag in tagged_tokens if tag not in ["NNP", "NNPS"]]

        # Join tokens back to string
        processed_text = " ".join(tokens)
        return processed_text


    def create_index_dir(self):
        """Create the index directory if it doesn't exist."""
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

    def create_or_open_index(self):
        """Create or open the index"""
        if not os.listdir(self.index_dir):
            self.ix = create_in(self.index_dir, self.schema)
        else:
            self.ix = open_dir(self.index_dir)


    def index_document(self, file_path: str, file_type: str, location: str,  content: str) -> None:
        """Index a document."""
        writer = self.ix.writer()
        processed_content = self.process_text(content)
        writer.add_document(
            path=file_path,
            filetype=file_type,
            location=location,
            content=processed_content,
            score=1.0
        )
        writer.commit()



    def index_pdf(self, file_path: str) -> tuple[bool, str]:
        """Index a PDF document."""
        try:
            pdf_reader = PyPDF2.PdfReader(file_path)
            indexed_pages = 0
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text:
                    processed_text = self.process_text(text)
                    self.index_document(
                        file_path=file_path,
                        file_type="pdf",
                        location=f"Page {page_num}",
                        content=processed_text
                    )

                    indexed_pages += 1

            if indexed_pages > 0:
                return True, f"Indexed {indexed_pages} pages from {file_path}"
            else:
                return False, f"No text found in {file_path}"
                
        except Exception as e:
            return False, f"Error indexing PDF {file_path}: {str(e)}"
        


    def index_txt(self, file_path: str) -> tuple[bool, str]:
        """Index a text document."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                processed_text = self.process_text(content)
                self.index_document(
                    file_path=file_path,
                    file_type="txt",
                    location="Full Document",
                    content=processed_text
                )
            return True, f"Indexed TXT: {file_path}"
        except Exception as e:
            return False, f"Error indexing text {file_path}: {str(e)}"
        

    def index_csv(self, file_path: str) -> tuple[bool, str]:
        """Index a CSV file, row by row."""
        try:
            with open(file_path, 'r', encoding="utf-8", errors='ignore') as file:
                reader = csv.reader(file)
                for row_num, row in enumerate(reader, 1):
                    for cell in row:
                        content = " ".join(cell for cell in row if cell)
                        location = f"Row {row_num}"
                        self.index_document(file_path, "csv", location, content)
                        indexed_rows += 1

            if indexed_rows > 0:
                return True, f"Indexed CSV: {file_path} ({indexed_rows} rows)"

            return False, "Empty CSV File"
        except Exception as e:
            return False, f"Error indexing CSV {file_path}: {str(e)}"
        

    def index_web(self, url: str) -> tuple[bool, str]:
        """Index content from a web page"""
        # send http request to the given url
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.text if soup.title else ""
        paragraphs = soup.find_all('p')

        indexed_items = 0
        if title.strip():
            self.index_document(url, "web", "Title", title)
            indexed_items +=1

        for i, p in enumerate(paragraphs, i):
            text = p.get_text().strip()
            if text:
                self.index_document(url, "web", f"Paragraph {i}", text)
                indexed_items += 1

        if indexed_items > 0:
            return True, f"Indexed Web Page: {url} ({indexed_items} sections)"
        
        return False, "Web page had no content"




