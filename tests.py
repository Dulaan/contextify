import unittest
from unittest.mock import patch, MagicMock
from funcs import extract_citations  # Replace with the actual module name

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
        expected_output = {"cite1": "Citation text"}  # Assuming the regex cleans up the text
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

        pdf_path = "error.pdf"
        result = extract_citations(pdf_path)
        
        # Expected to return an empty dictionary due to the exception
        self.assertEqual(result, {})

        # Verify that the exception was raised and handled
        mock_PdfReader.assert_called_once_with(pdf_path)

if __name__ == "__main__":
    unittest.main()
