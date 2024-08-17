import os
import re
import requests
from typing import Dict, List, Tuple, Optional
import logging

import pymupdf
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Text
from serpapi.google_scholar_search import GoogleScholarSearch
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MAX_TOKENS = 1024
TEMPERATURE = 0.5

def extract_citations(pdf_path: str) -> Dict[str, str]:
    """Extract citations from a PDF file."""
    cites = {}
    try:
        reader = PdfReader(pdf_path)
        doc = pymupdf.open(pdf_path)
        if not reader.pages:
            logger.warning(f"PDF file {pdf_path} is empty.")
            return cites

        height = reader.pages[0].mediabox.height

        for cite, info in reader.named_destinations.items():
            if cite.startswith("cite"):
                x1, y1 = info["/Left"], height - info['/Top']
                rect = pymupdf.Rect(x1, y1, x1 + 400, y1 + 20)
                page_index = next((i for i, page in enumerate(reader.pages) if page == info["/Page"]), None)
                if page_index is not None:
                    cites[cite] = doc[page_index].get_textbox(rect)
    except Exception as e:
        logger.error(f"Error extracting citations from {pdf_path}: {str(e)}")
    return cites

def create_folder(pdf_path: str) -> str:
    """Create a folder for storing reference documents."""
    title = re.sub(r"\W+", "", pdf_path[:-4]) + "_refs"
    try:
        os.makedirs(title, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating folder {title}: {str(e)}")
        raise
    return title

def download_document(folder: str, title: str, cite: str) -> Optional[str]:
    """Download a document from Google Scholar."""
    api_key = os.environ.get("SERP_API_KEY")
    if not api_key:
        logger.error("SERP API key not found in environment variables.")
        return None

    filename = re.sub(r"\W+", "", cite[5:])

    try:
        search = GoogleScholarSearch({"q": title, "api_key": api_key})
        results = search.get_dict().get("organic_results", [])
        if not results:
            logger.warning(f"No results found for {title}")
            return None

        url = results[0].get("resources", [{}])[0].get("link")
        if not url:
            logger.warning(f"No download link found for {title}")
            return None

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        file_path = os.path.join("/content", folder, f"{filename}.pdf")
        with open(file_path, "wb") as f:
            f.write(response.content)

        return f"{filename}.pdf"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading document {title}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while downloading {title}: {str(e)}")

    return None

def download_all_documents(folder: str, cites: Dict[str, str]) -> None:
    """Download all cited documents."""
    for cite, title in cites.items():
        download_document(folder, title, cite)

def extract_text(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() for page in reader.pages)
        return re.sub("\n", " ", text)
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""

def create_summary_prompt(text: str, context: str) -> str:
    """Create a prompt for summarizing a passage with context."""
    return f'Explain the following passage: "{text}" using this context from the paper cited: "{context}". CONTEXTUALIZE: '

def summarize_text(client: Groq, text: str) -> Optional[str]:
    """Summarize text using the Groq API."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional summarizer hired to provide context to citations within papers. Use the provided text chunks to explain the citation's relevance to the passage concisely.",
                },
                {"role": "user", "content": text},
            ],
            model="llama3-70b-8192",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=1,
            stop=None,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return None

def summarize_paper_with_context(client: Groq, text: str, ref_ctx: str) -> Optional[str]:
    """Summarize a paper passage with reference context."""
    prompt = create_summary_prompt(text, ref_ctx)
    return summarize_text(client, prompt)

def extract_citation_locations(pdf_path: str, cites: Dict[str, str]) -> Dict[str, Dict[int, List[List[float]]]]:
    """Extract citation locations from a PDF file."""
    locs = {cite: {} for cite in cites}
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            if "/Annots" in page:
                for annot in page["/Annots"]:
                    obj = annot.get_object()
                    if (
                        obj.get("/Subtype") == "/Link"
                        and obj.get("/A", {}).get("/S") == "/GoTo"
                    ):
                        cite_ref = obj.get("/A", {}).get("/D")
                        if cite_ref in locs:
                            if page_num not in locs[cite_ref]:
                                locs[cite_ref][page_num] = []
                            locs[cite_ref][page_num].append(obj.get("/Rect", []))
    except Exception as e:
        logger.error(f"Error extracting citation locations from {pdf_path}: {str(e)}")

    return locs

def extract_citation_context(pdf_path: str, cites: Dict[str, str]) -> Dict[str, Dict[int, List[str]]]:
    """Extract citation context from a PDF file."""
    cite_dict = {cite: {} for cite in cites}
    try:
        doc = pymupdf.open(pdf_path)
        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages):
            if "/Annots" in page:
                for annot in page["/Annots"]:
                    obj = annot.get_object()
                    if (
                        obj.get("/Subtype") == "/Link"
                        and obj.get("/A", {}).get("/S") == "/GoTo"
                    ):
                        cite_ref = obj.get("/A", {}).get("/D")
                        if cite_ref in cites:
                            info = obj.get("/Rect", [0, 0, 0, 0])
                            rect = pymupdf.Rect(100, 792 - info[1] - 10, 500, 792 - info[1] + 15)
                            text = re.sub("\n", " ", doc[page_num].get_textbox(rect))
                            if page_num not in cite_dict[cite_ref]:
                                cite_dict[cite_ref][page_num] = []
                            cite_dict[cite_ref][page_num].append(text)
    except Exception as e:
        logger.error(f"Error extracting citation context from {pdf_path}: {str(e)}")

    return cite_dict

def initialize_retriever(folder: str) -> Optional[FAISS]:
    """Initialize a FAISS retriever for document search."""
    try:
        docs = [extract_text(os.path.join(folder, file)) for file in os.listdir(folder) if file.endswith('.pdf')]
        if not docs:
            logger.warning(f"No PDF documents found in {folder}")
            return None

        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator=" ")
        texts = text_splitter.create_documents(docs)
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        return db.as_retriever()
    except Exception as e:
        logger.error(f"Error initializing retriever for {folder}: {str(e)}")
        return None

def generate_summaries(folder_path: str, ctx: Dict[str, Dict[int, List[str]]]) -> Dict[str, Dict[int, List[str]]]:
    """Generate summaries for citation contexts."""
    summaries = {}
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    if not client.api_key:
        logger.error("GROQ API key not found in environment variables.")
        return summaries

    retriever = initialize_retriever(folder_path)
    if not retriever:
        logger.error("Failed to initialize retriever.")
        return summaries

    for cite, pages in ctx.items():
        summaries[cite] = {}
        for page, contexts in pages.items():
            summaries[cite][page] = []
            for context in contexts:
                try:
                    docs = retriever.get_relevant_documents(context, k=2)
                    info = str([doc.page_content for doc in docs])
                    summary = summarize_paper_with_context(client, context, info)
                    if summary:
                        summaries[cite][page].append(summary)
                        logger.info(f"Generated summary for {cite} on page {page}")
                    else:
                        logger.warning(f"Failed to generate summary for {cite} on page {page}")
                except Exception as e:
                    logger.error(f"Error generating summary for {cite} on page {page}: {str(e)}")

    return summaries

def add_annotations(pdf_path: str, cites: Dict[str, str], locations: Dict[str, Dict[int, List[List[float]]]], summaries: Dict[str, Dict[int, List[str]]]) -> None:
    """Add annotations to a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        for page in reader.pages:
            writer.add_page(page)

        for cite in cites:
            for page, coords in locations.get(cite, {}).items():
                for i, rect in enumerate(coords):
                    if page in summaries.get(cite, {}) and i < len(summaries[cite][page]):
                        annotation = Text(text=re.sub("\n", "", summaries[cite][page][i]), rect=rect)
                        writer.add_annotation(page_number=page, annotation=annotation)

        with open("annotated.pdf", "wb") as fp:
            writer.write(fp)
        logger.info("Successfully created annotated PDF.")
    except Exception as e:
        logger.error(f"Error adding annotations to {pdf_path}: {str(e)}")

# Main execution flow can be organized here