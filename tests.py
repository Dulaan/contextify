import unittest
from unittest.mock import patch, MagicMock
from funcs import extract_citations, download_document, extract_text
import logging


class TestExtractCitations(unittest.TestCase):
    @patch("funcs.PdfReader")  # Mock PdfReader
    @patch("funcs.pymupdf.open")  # Mock pymupdf.open
    def test_extract_citations_normal_case(self, mock_pymupdf_open, mock_PdfReader):
        """Test normal case with valid citations."""
        # Mock the PdfReader instance and its properties
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock(mediabox=MagicMock(height=800))]
        mock_reader.named_destinations = {
            "cite1": {"/Left": 100, "/Top": 200, "/Page": mock_reader.pages[0]}
        }
        mock_PdfReader.return_value = mock_reader

        # Mock the pymupdf.open instance and its method
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_textbox.return_value = "Citation text"
        mock_doc.__getitem__.return_value = mock_page
        mock_pymupdf_open.return_value = mock_doc

        # Call the function and check the result
        pdf_path = "dummy.pdf"
        expected_output = {
            "cite1": "Citation text"
        }  # Assuming the regex cleans up the text
        result = extract_citations(pdf_path)

        self.assertEqual(result, expected_output)

        # Verify interactions
        mock_PdfReader.assert_called_once_with(pdf_path)
        mock_pymupdf_open.assert_called_once_with(pdf_path)
        mock_page.get_textbox.assert_called_once()

    @patch("funcs.PdfReader")
    @patch("funcs.pymupdf.open")
    def test_no_citations(self, mock_pymupdf_open, mock_PdfReader):
        """Test case where the PDF has pages but no named destinations for citations."""
        # Mock the PdfReader instance with pages but no named destinations
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock(mediabox=MagicMock(height=800))]
        mock_reader.named_destinations = {}
        mock_PdfReader.return_value = mock_reader

        # Mock pymupdf.open as well
        mock_pymupdf_open.return_value = MagicMock()

        pdf_path = "no_citations.pdf"
        result = extract_citations(pdf_path)

        # Expected to return an empty dictionary since there are no citations
        self.assertEqual(result, {})

        # Verify interactions
        mock_PdfReader.assert_called_once_with(pdf_path)
        mock_pymupdf_open.assert_called_once_with(pdf_path)

    @patch("funcs.PdfReader")
    @patch("funcs.pymupdf.open")
    def test_exception_handling(self, mock_pymupdf_open, mock_PdfReader):
        """Test case where an exception occurs during PDF processing."""
        # Mock the PdfReader to raise an exception
        mock_PdfReader.side_effect = Exception("Test exception")

        mock_pymupdf_open.return_value = MagicMock()

        pdf_path = "error.pdf"
        result = extract_citations(pdf_path)

        # Expected to return an empty dictionary due to the exception
        self.assertEqual(result, {})

        # Verify that the exception was raised and handled
        mock_PdfReader.assert_called_once_with(pdf_path)


class TestDownloadDocument(unittest.TestCase):
    @patch("funcs.requests.get")  # Mock requests.get
    @patch("funcs.os.path.join")  # Mock os.path.join
    @patch("builtins.open")
    def test_download_document_success(
        self, mock_open, mock_path_join, mock_requests_get
    ):
        """Test a successful document download."""
        # Mock the client search response
        mock_open = MagicMock()
        mock_open.write.return_value = "example_doc"
        mock_client = MagicMock()
        mock_client.search.return_value.as_dict.return_value = {
            "organic_results": [{"resources": [{"link": "http://example.com/doc.pdf"}]}]
        }

        # Mock the requests.get response
        mock_response = MagicMock()
        mock_response.content = b"PDF content"
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        # Mock the os.path.join behavior
        mock_path_join.return_value = "/mock_folder/mock_filename.pdf"

        folder = "/mock_folder"
        title = "Sample Document"
        cite = "cite:example_citation"

        result = download_document(folder, title, cite, mock_client)

        # Assert the expected filename is returned
        self.assertEqual(result, "example_citation.pdf")

        # Verify interactions
        mock_client.search.assert_called_once_with(
            {"engine": "google_scholar", "q": title}
        )
        mock_requests_get.assert_called_once_with(
            "http://example.com/doc.pdf", timeout=30
        )
        mock_path_join.assert_called_once_with(folder, "example_citation.pdf")

    @patch("funcs.requests.get")
    def test_download_document_no_results(
        self, mock_requests_get
    ):
        """Test when no results are returned from the search."""
        # Mock the client search response to return no results
        mock_client = MagicMock()
        mock_client.search.return_value.as_dict.return_value = {"organic_results": []}

        folder = "/mock_folder"
        title = "Nonexistent Document"
        cite = "cite:no_results"

        result = download_document(folder, title, cite, mock_client)

        # Assert that None is returned because no results were found
        self.assertIsNone(result)

        # Verify interactions
        mock_client.search.assert_called_once_with(
            {"engine": "google_scholar", "q": title}
        )
        mock_requests_get.assert_not_called()  # No download should happen


    def test_download_document_exception(self):
        """Test when an exception occurs during the download process."""
        # Mock the client search response to return an error.
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Test exception")

        folder = "/mock_folder"
        title = "Error Document"
        cite = "cite:error"

        result = download_document(folder, title, cite, mock_client)

        self.assertIsNone(result)


class TestExtractText(unittest.TestCase):
    @patch("funcs.PdfReader")
    def test_extract_text(self, mock_PdfReader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "example page"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_PdfReader.return_value = mock_reader

        file_path = "example.pdf"
        result = extract_text(file_path)
        mock_PdfReader.assert_called_once_with(file_path)
        self.assertEqual(result, "example page")

if __name__ == "__main__":
    unittest.main()
