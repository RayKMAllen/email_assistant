"""
Comprehensive unit tests for the EmailLLMProcessor (assistant core functionality).
"""

import pytest
import json
from unittest.mock import patch, MagicMock, Mock
from botocore.exceptions import ClientError, NoCredentialsError

from src.assistant.llm_session import EmailLLMProcessor
from src.assistant import utils


class TestEmailLLMProcessor:
    """Test the EmailLLMProcessor class"""
    
    @pytest.fixture
    def mock_boto_clients(self):
        """Mock boto3 clients for testing"""
        with patch("src.assistant.llm_session.boto3.client") as mock_boto:
            mock_s3_client = MagicMock()
            mock_bedrock_runtime = MagicMock()
            mock_boto.side_effect = [mock_s3_client, mock_bedrock_runtime]
            yield mock_s3_client, mock_bedrock_runtime
    
    @pytest.fixture
    def session(self, mock_boto_clients):
        """Create EmailLLMProcessor instance with mocked clients"""
        return EmailLLMProcessor()
    
    def test_initialization(self, session, mock_boto_clients):
        """Test EmailLLMProcessor initialization"""
        mock_client, mock_runtime = mock_boto_clients
        
        assert session.client == mock_client
        assert session.runtime == mock_runtime
        assert session.text is None
        assert session.key_info is None
        assert session.last_draft is None
        assert session.history == []
    
    def test_send_prompt_success(self, session):
        """Test successful prompt sending"""
        session.runtime.invoke_model = MagicMock()
        fake_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps({"content": [{"text": "model output"}]}).encode("utf-8")
                )
            )
        }
        session.runtime.invoke_model.return_value = fake_response
        
        result = session.send_prompt("test prompt")
        
        assert result == "model output"
        assert len(session.history) == 2
        assert session.history[-2]["content"] == "test prompt"
        assert session.history[-2]["role"] == "user"
        assert session.history[-1]["content"] == "model output"
        assert session.history[-1]["role"] == "assistant"
    
    def test_extract_key_info_success(self, session):
        """Test successful key information extraction"""
        session.text = "email text"
        session.send_prompt = MagicMock(return_value=json.dumps({"summary": "test summary"}))
        
        session.extract_key_info()
        
        assert session.key_info == {"summary": "test summary"}
        session.send_prompt.assert_called_once()
    
    def test_extract_key_info_json_decode_error(self, session):
        """Test key info extraction with JSON decode error"""
        session.text = "email text"
        session.send_prompt = MagicMock(return_value="not valid json")
        
        with pytest.raises(Exception, match="Failed to parse key information"):
            session.extract_key_info()
    
    def test_draft_reply_success(self, session):
        """Test successful reply drafting"""
        session.text = "original email"
        session.key_info = {"summary": "meeting request"}
        session.send_prompt = MagicMock(return_value="drafted reply")
        
        result = session.draft_reply(tone="formal")
        
        assert result == "drafted reply"
        assert session.last_draft == "drafted reply"
        session.send_prompt.assert_called_once()
    
    def test_refine_with_existing_draft(self, session):
        """Test refining an existing draft"""
        session.last_draft = "original draft"
        session.key_info = {"summary": "meeting request"}
        session.send_prompt = MagicMock(return_value="refined draft")
        
        result = session.refine("make it more formal")
        
        assert result == "refined draft"
        assert session.last_draft == "refined draft"
    
    def test_save_draft_success(self, session):
        """Test successful draft saving"""
        session.last_draft = "draft to save"
        
        with patch('src.assistant.llm_session.save_draft_to_file') as mock_save:
            session.save_draft("test.txt")
            mock_save.assert_called_once_with("draft to save", "test.txt")


# Tests for utility functions
class TestUtilityFunctions:
    """Test utility functions used by the assistant"""
    
    def test_process_path_or_email_reads_file(self, tmp_path):
        """Test processing an existing file"""
        file = tmp_path / "mail.txt"
        file.write_text("hello")
        result = utils.process_path_or_email(str(file))
        assert result == "hello"
    
    def test_process_path_or_email_returns_text(self):
        """Test processing raw text content"""
        text = "This is not a file path"
        result = utils.process_path_or_email(text)
        assert result == text
    
    def test_save_draft_to_file(self, tmp_path):
        """Test saving draft to file"""
        file = tmp_path / "out.txt"
        utils.save_draft_to_file("draft", str(file))
        assert file.read_text() == "draft"
