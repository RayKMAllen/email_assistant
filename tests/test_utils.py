"""
Comprehensive unit tests for utility functions.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from botocore.exceptions import NoCredentialsError, ClientError

from src.assistant.utils import (
    process_path_or_email,
    extract_text_from_pdf,
    extract_text,
    make_now_filename,
    save_draft_to_file,
    save_draft_to_s3
)


class TestProcessPathOrEmail:
    """Test the process_path_or_email function"""
    
    def test_process_existing_file(self, tmp_path):
        """Test processing an existing file"""
        # Create a temporary file
        test_file = tmp_path / "test_email.txt"
        test_content = "This is a test email content"
        test_file.write_text(test_content)
        
        with patch('src.assistant.utils.extract_text') as mock_extract:
            mock_extract.return_value = test_content
            
            result = process_path_or_email(str(test_file))
            
            assert result == test_content
            mock_extract.assert_called_once_with(str(test_file))
    
    def test_process_non_existing_file(self, capsys):
        """Test processing non-existing file (should treat as raw content)"""
        non_existing_path = "/path/that/does/not/exist.txt"
        
        result = process_path_or_email(non_existing_path)
        
        assert result == non_existing_path
        captured = capsys.readouterr()
        assert "File not found, assuming input is raw email content." in captured.out
    
    def test_process_raw_email_content(self, capsys):
        """Test processing raw email content"""
        email_content = """From: john@example.com
To: jane@example.com
Subject: Test Email

Hello Jane,

This is a test email.

Best regards,
John"""
        
        result = process_path_or_email(email_content)
        
        assert result == email_content
        captured = capsys.readouterr()
        assert "File not found, assuming input is raw email content." in captured.out


class TestExtractTextFromPdf:
    """Test the extract_text_from_pdf function"""
    
    @patch('src.assistant.utils.pymupdf.open')
    def test_extract_text_from_pdf_success(self, mock_open):
        """Test successful PDF text extraction"""
        # Mock PyMuPDF document and pages
        mock_doc = Mock()
        mock_page1 = Mock()
        mock_page2 = Mock()
        mock_page1.get_text.return_value = "Page 1 content\n"
        mock_page2.get_text.return_value = "Page 2 content\n"
        mock_doc.__iter__ = Mock(return_value=iter([mock_page1, mock_page2]))
        mock_doc.close = Mock()
        mock_open.return_value = mock_doc
        
        result = extract_text_from_pdf("test.pdf")
        
        assert result == "Page 1 content\nPage 2 content\n"
        mock_open.assert_called_once_with("test.pdf")
        mock_doc.close.assert_called_once()
    
    @patch('src.assistant.utils.pymupdf.open')
    def test_extract_text_from_pdf_empty_document(self, mock_open):
        """Test PDF text extraction from empty document"""
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([]))
        mock_doc.close = Mock()
        mock_open.return_value = mock_doc
        
        result = extract_text_from_pdf("empty.pdf")
        
        assert result == ""
        mock_doc.close.assert_called_once()
    
    @patch('src.assistant.utils.pymupdf.open')
    def test_extract_text_from_pdf_exception(self, mock_open):
        """Test PDF text extraction with exception"""
        mock_open.side_effect = Exception("PDF error")
        
        with pytest.raises(Exception, match="PDF error"):
            extract_text_from_pdf("bad.pdf")


class TestExtractText:
    """Test the extract_text function"""
    
    @patch('src.assistant.utils.extract_text_from_pdf')
    def test_extract_text_pdf_file(self, mock_pdf_extract, capsys):
        """Test extracting text from PDF file"""
        mock_pdf_extract.return_value = "PDF content"
        
        result = extract_text("document.pdf")
        
        assert result == "PDF content"
        mock_pdf_extract.assert_called_once_with("document.pdf")
        captured = capsys.readouterr()
        assert "Extracting text from PDF..." in captured.out
    
    def test_extract_text_regular_file(self, tmp_path, capsys):
        """Test extracting text from regular text file"""
        test_file = tmp_path / "test.txt"
        test_content = "This is test content\nWith multiple lines"
        test_file.write_text(test_content)
        
        result = extract_text(str(test_file))
        
        assert result == test_content
        captured = capsys.readouterr()
        assert "Extracting text from non-PDF file..." in captured.out
    
    def test_extract_text_file_not_found(self):
        """Test extracting text from non-existent file"""
        with pytest.raises(FileNotFoundError):
            extract_text("nonexistent.txt")
    
    def test_extract_text_permission_error(self, tmp_path):
        """Test extracting text with permission error"""
        test_file = tmp_path / "restricted.txt"
        test_file.write_text("content")
        
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError, match="Access denied"):
                extract_text(str(test_file))


class TestMakeNowFilename:
    """Test the make_now_filename function"""
    
    @patch('src.assistant.utils.datetime')
    def test_make_now_filename_format(self, mock_datetime):
        """Test filename format generation"""
        mock_now = Mock()
        mock_now.strftime.return_value = "20231201_143022"
        mock_datetime.datetime.now.return_value = mock_now
        
        result = make_now_filename()
        
        assert result == "draft_20231201_143022.txt"
        mock_now.strftime.assert_called_once_with('%Y%m%d_%H%M%S')
    
    def test_make_now_filename_real_datetime(self):
        """Test filename generation with real datetime"""
        result = make_now_filename()
        
        # Should match the pattern draft_YYYYMMDD_HHMMSS.txt
        assert result.startswith("draft_")
        assert result.endswith(".txt")
        assert len(result) == len("draft_20231201_143022.txt")
        
        # Extract the timestamp part and verify it's valid
        timestamp_part = result[6:-4]  # Remove "draft_" and ".txt"
        assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS
        assert timestamp_part[8] == "_"  # Underscore separator
        
        # Should be able to parse as datetime
        datetime.strptime(timestamp_part, '%Y%m%d_%H%M%S')


class TestSaveDraftToFile:
    """Test the save_draft_to_file function"""
    
    def test_save_draft_with_filepath(self, tmp_path, capsys):
        """Test saving draft to specified filepath"""
        draft_content = "This is a draft email"
        output_file = tmp_path / "my_draft.txt"
        
        save_draft_to_file(draft_content, str(output_file))
        
        assert output_file.exists()
        assert output_file.read_text() == draft_content
        captured = capsys.readouterr()
        assert f"Saving draft to {output_file}..." in captured.out
    
    @patch('src.assistant.utils.os.path.expanduser')
    @patch('src.assistant.utils.make_now_filename')
    def test_save_draft_default_location(self, mock_filename, mock_expanduser, tmp_path, capsys):
        """Test saving draft to default location"""
        draft_content = "Default location draft"
        mock_expanduser.return_value = str(tmp_path)
        mock_filename.return_value = "draft_20231201_143022.txt"
        
        save_draft_to_file(draft_content)
        
        # Check that drafts directory was created
        drafts_dir = tmp_path / "drafts"
        assert drafts_dir.exists()
        assert drafts_dir.is_dir()
        
        # Check that file was created with correct content
        expected_file = drafts_dir / "draft_20231201_143022.txt"
        assert expected_file.exists()
        assert expected_file.read_text() == draft_content
        
        captured = capsys.readouterr()
        assert f"Saving draft to {expected_file}..." in captured.out
    
    def test_save_draft_create_directories(self, tmp_path, capsys):
        """Test that save_draft_to_file handles nested paths when directories exist"""
        draft_content = "Draft with nested path"
        nested_path = tmp_path / "level1" / "level2" / "draft.txt"
        
        # Create the directories first since save_draft_to_file doesn't create them for custom paths
        nested_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_draft_to_file(draft_content, str(nested_path))
        
        # Directory should exist and file should exist
        assert nested_path.parent.exists()
        assert nested_path.exists()
        assert nested_path.read_text() == draft_content
    
    def test_save_draft_permission_error(self, tmp_path):
        """Test handling permission error when saving draft"""
        draft_content = "Permission test draft"
        output_file = tmp_path / "readonly_draft.txt"
        
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="Permission denied"):
                save_draft_to_file(draft_content, str(output_file))


class TestSaveDraftToS3:
    """Test the save_draft_to_s3 function"""
    
    @patch('src.assistant.utils.boto3.client')
    def test_save_draft_to_s3_success(self, mock_boto_client, capsys):
        """Test successful S3 draft saving"""
        mock_s3 = Mock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.return_value = None  # Bucket exists
        mock_s3.put_object.return_value = None
        
        draft_content = "S3 draft content"
        bucket_name = "test-bucket"
        filepath = "drafts/test_draft.txt"
        
        save_draft_to_s3(draft_content, bucket_name, filepath)
        
        mock_boto_client.assert_called_once_with("s3")
        mock_s3.head_bucket.assert_called_once_with(Bucket=bucket_name)
        mock_s3.put_object.assert_called_once_with(
            Bucket=bucket_name,
            Key=filepath,
            Body=draft_content.encode("utf-8")
        )
        
        captured = capsys.readouterr()
        assert f"Attempting to save draft to S3 bucket: {bucket_name}" in captured.out
        assert f"S3 key will be: {filepath}" in captured.out
        assert f"Draft saved successfully to s3://{bucket_name}/{filepath}" in captured.out
    
    @patch('src.assistant.utils.boto3.client')
    @patch('src.assistant.utils.make_now_filename')
    def test_save_draft_to_s3_default_filepath(self, mock_filename, mock_boto_client, capsys):
        """Test S3 draft saving with default filepath"""
        mock_s3 = Mock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.return_value = None
        mock_s3.put_object.return_value = None
        mock_filename.return_value = "draft_20231201_143022.txt"
        
        draft_content = "Default filepath draft"
        bucket_name = "test-bucket"
        
        save_draft_to_s3(draft_content, bucket_name)
        
        expected_key = "drafts/draft_20231201_143022.txt"
        mock_s3.put_object.assert_called_once_with(
            Bucket=bucket_name,
            Key=expected_key,
            Body=draft_content.encode("utf-8")
        )
    
    @patch('src.assistant.utils.boto3.client')
    def test_save_draft_to_s3_bucket_not_accessible(self, mock_boto_client, capsys):
        """Test S3 draft saving when bucket is not accessible"""
        mock_s3 = Mock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.side_effect = ClientError(
            {'Error': {'Code': 'NoSuchBucket'}}, 'HeadBucket'
        )
        mock_s3.put_object.return_value = None
        
        draft_content = "Bucket warning test"
        bucket_name = "nonexistent-bucket"
        
        save_draft_to_s3(draft_content, bucket_name, "test.txt")
        
        captured = capsys.readouterr()
        assert f"Warning: Cannot access bucket {bucket_name}" in captured.out
        # Should still attempt to save
        mock_s3.put_object.assert_called_once()
    
    @patch('src.assistant.utils.boto3.client')
    def test_save_draft_to_s3_no_credentials(self, mock_boto_client):
        """Test S3 draft saving with no credentials"""
        mock_boto_client.side_effect = NoCredentialsError()
        
        draft_content = "No credentials test"
        bucket_name = "test-bucket"
        
        with pytest.raises(Exception, match="Failed to save draft to S3"):
            save_draft_to_s3(draft_content, bucket_name, "test.txt")
    
    @patch('src.assistant.utils.boto3.client')
    def test_save_draft_to_s3_access_denied(self, mock_boto_client, capsys):
        """Test S3 draft saving with access denied"""
        mock_s3 = Mock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.return_value = None
        mock_s3.put_object.side_effect = ClientError(
            {'Error': {'Code': 'AccessDenied'}}, 'PutObject'
        )
        
        draft_content = "Access denied test"
        bucket_name = "test-bucket"
        
        with pytest.raises(Exception, match="Failed to save draft to S3"):
            save_draft_to_s3(draft_content, bucket_name, "test.txt")
    
    @patch('src.assistant.utils.boto3.client')
    def test_save_draft_to_s3_no_such_bucket(self, mock_boto_client, capsys):
        """Test S3 draft saving with non-existent bucket"""
        mock_s3 = Mock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.return_value = None
        mock_s3.put_object.side_effect = ClientError(
            {'Error': {'Code': 'NoSuchBucket'}}, 'PutObject'
        )
        
        draft_content = "No such bucket test"
        bucket_name = "nonexistent-bucket"
        
        with pytest.raises(Exception, match="Failed to save draft to S3"):
            save_draft_to_s3(draft_content, bucket_name, "test.txt")


# Integration tests combining multiple functions
class TestUtilsIntegration:
    """Integration tests for utility functions"""
    
    def test_full_file_processing_workflow(self, tmp_path):
        """Test complete workflow from file to processed content"""
        # Create test email file
        email_content = """From: test@example.com
To: user@example.com
Subject: Test Email

Hello,

This is a test email for processing.

Best regards,
Test User"""
        
        test_file = tmp_path / "test_email.txt"
        test_file.write_text(email_content)
        
        # Process the file
        result = process_path_or_email(str(test_file))
        
        assert result == email_content
    
    def test_draft_saving_workflow(self, tmp_path):
        """Test complete draft saving workflow"""
        draft_content = "This is a complete draft email for testing the workflow."
        
        # Test local saving
        output_file = tmp_path / "workflow_test.txt"
        save_draft_to_file(draft_content, str(output_file))
        
        assert output_file.exists()
        assert output_file.read_text() == draft_content
        
        # Test filename generation
        filename = make_now_filename()
        assert filename.startswith("draft_")
        assert filename.endswith(".txt")