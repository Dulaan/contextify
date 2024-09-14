import os
import re
import requests
from typing import Dict, List, Tuple, Optional, Any
import logging
import torch
import io
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
import pymupdf
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Text
import time
from groq import Groq


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MAX_TOKENS = 1024
TEMPERATURE = 0.5
CHUNK_N = 2

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
                x1, y1 = info["/Left"], height - info["/Top"]
                rect = pymupdf.Rect(x1, y1, x1 + 400, y1 + 20)
                page_index = next(
                    (i for i, page in enumerate(reader.pages) if page == info["/Page"]),
                    None,
                )
                if page_index is not None:
                    cites[cite] = doc[page_index].get_textbox(rect)
    except Exception as e:
        logger.error(f"Error extracting citations from {pdf_path}: {str(e)}")
        raise
    return cites


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
        raise


def summarize_paper_with_context(
    client: Groq, text: str, ref_ctx: str
) -> Optional[str]:
    """Summarize a paper passage with reference context."""
    prompt = create_summary_prompt(text, ref_ctx)
    return summarize_text(client, prompt)


def extract_citation_locations(
    pdf_path: str, cites: Dict[str, str]
) -> Dict[str, Dict[int, List[List[float]]]]:
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
        raise

    return locs


def extract_citation_context(
    pdf_path: str, cites: Dict[str, str]
) -> Dict[str, Dict[int, List[str]]]:
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
                            rect = pymupdf.Rect(
                                100, 792 - info[1] - 10, 500, 792 - info[1] + 15
                            )
                            text = re.sub("\n", " ", doc[page_num].get_textbox(rect))
                            if page_num not in cite_dict[cite_ref]:
                                cite_dict[cite_ref][page_num] = []
                            cite_dict[cite_ref][page_num].append(text)
    except Exception as e:
        logger.error(f"Error extracting citation context from {pdf_path}: {str(e)}")
        raise

    return cite_dict


def chunk(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    gap = size - overlap
    i = 0
    while i + gap <= len(text):
        chunks.append(text[i : i + gap])
        i += gap
    if i + gap > len(text):
        chunks.append(text[i:])
    return chunks


def getPdf(response_data):
    for result in response_data["webPages"]["value"]:
        if "pdf" in result["url"]:
            return result["url"]
    return None


def download_pdf(
    title: str, headers: Dict[str, str], search_url: str, params: Dict[str, Any]
) -> bytes:
    params["q"] = title
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    pdf_link = getPdf(search_results)
    if not pdf_link:
        raise ValueError(f"No PDF link found for title: {title}")

    file_response = requests.get(pdf_link, timeout=30)
    file_response.raise_for_status()
    return file_response.content


def process_pdf(pdf_content: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_content))
    text = " ".join(page.extract_text() for page in reader.pages)
    text = re.sub("\n", " ", text)
    return chunk(text, CHUNK_SIZE, CHUNK_OVERLAP)


def download_and_retrieve(
    cites: Dict[str, str]
) -> Tuple[Dict[str, torch.FloatTensor], Dict[str, List[str]]]:
    key = os.environ.get("BING_API_KEY")
    if not key:
        raise ValueError("BING_API_KEY not found in environment variables")

    headers = {"Ocp-Apim-Subscription-Key": key}
    params = {"textDecorations": True, "textFormat": "HTML"}
    search_url = "https://api.bing.microsoft.com/v7.0/search"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeds = {}
    texts = {}

    for cite, title in cites.items():
        try:
            pdf_content = download_pdf(title, headers, search_url, params)
            chunks = process_pdf(pdf_content)
            doc_embeds = torch.FloatTensor(model.encode(chunks))
            embeds[cite] = doc_embeds
            texts[cite] = chunks
        except Exception as e:
            logger.error(f"Error processing citation {cite}: {str(e)}")

    return embeds, texts


def n_generate_summaries(
    ctx: Dict[str, Dict[int, List[str]]],
    embeds: Dict[str, torch.FloatTensor],
    texts: Dict[str, List[str]],
) -> Dict[str, Dict[int, List[str]]]:
    summaries = {}
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    if not client.api_key:
        logger.error("GROQ API key not found in environment variables.")
        return summaries

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for cite, pages in ctx.items():
        doc_embeds = embeds.get(cite)
        if doc_embeds is None:
            logger.warning(f"No embeddings found for citation {cite}")
            continue

        summaries[cite] = {}
        for page, contexts in pages.items():
            summaries[cite][page] = []
            for context in contexts:
                try:
                    ctx_embeds = model.encode(context)
                    hits = semantic_search(ctx_embeds, doc_embeds, top_k = CHUNK_N)
                    info = str(
                        [
                            texts[cite][hits[0][i]["corpus_id"]]
                            for i in range(len(hits[0]))
                        ]
                    )
                    summary = summarize_paper_with_context(client, context, info)
                    if summary:
                        summaries[cite][page].append(summary)
                        logger.info(f"Generated summary for {cite} on page {page}")
                    else:
                        logger.warning(
                            f"Failed to generate summary for {cite} on page {page}"
                        )
                    time.sleep(3)
                except Exception as e:
                    logger.error(
                        f"Error generating summary for {cite} on page {page}: {str(e)}"
                    )
                    raise

    return summaries


def add_annotations(
    pdf_path: str,
    cites: Dict[str, str],
    locations: Dict[str, Dict[int, List[List[float]]]],
    summaries: Dict[str, Dict[int, List[str]]],
) -> Optional[str]:
    """Add annotations to a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        for page in reader.pages:
            writer.add_page(page)

        for cite in cites:
            for page, coords in locations.get(cite, {}).items():
                for i, rect in enumerate(coords):
                    if page in summaries.get(cite, {}) and i < len(
                        summaries[cite][page]
                    ):
                        annotation = Text(
                            text=re.sub("\n", "", summaries[cite][page][i]), rect=rect
                        )
                        writer.add_annotation(page_number=page, annotation=annotation)
        with open("annotated.pdf", "wb") as fp:
            writer.write(fp)
        logger.info("Successfully created annotated PDF.")
        return "annotated.pdf"
    except Exception as e:
        logger.error(f"Error adding annotations to {pdf_path}: {str(e)}")
        raise


def delete_folder(folder_path):
    for file in os.listdir(folder_path):
        os.remove(folder_path + "/" + file)
    os.rmdir(folder_path)
