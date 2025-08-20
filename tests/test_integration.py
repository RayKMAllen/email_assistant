"""
Integration tests for the email assistant system.
Tests complete workflows and component interactions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os

from src.assistant.conversational_agent import ConversationalEmailAgent
from src.assistant.conversation_state import ConversationState
from src.cli.cli import cli
from click.testing import CliRunner


class TestEmailProcessingWorkflow:
    """Test complete email processing workflows"""
    
    @pytest.fixture
    def mock_llm_responses(self):
        """Mock LLM responses for testing"""
        return {
            'extract_info': json.dumps({
                'summary': 'Meeting request from John Doe',
                'sender_name': 'John Doe',
                'sender_email': 'john@example.com',
                'receiver_name': 'Jane Smith',
                'subject': 'Weekly Team Meeting',
                'key_points': ['Schedule meeting', 'Discuss project status']
            }),
            'draft_reply': 'Dear John,\n\nThank you for your email about the weekly team meeting.\n\nI am available for the meeting as proposed. Please send me the calendar invite.\n\nBest regards,\nJane',
            'refined_reply': 'Dear John,\n\nThank you for your email regarding the weekly team meeting.\n\nI would be delighted to attend the meeting as proposed. Please send me the calendar invitation at your earliest convenience.\n\nBest regards,\nJane Smith'
        }
    
    @pytest.fixture
    def sample_email(self):
        """Sample email content for testing"""
        return """From: john@example.com
To: jane@example.com
Subject: Weekly Team Meeting

Dear Jane,

I hope this email finds you well. I wanted to reach out to schedule our weekly team meeting for next Tuesday at 2 PM.

The agenda will include:
- Project status updates
- Budget review
- Next quarter planning

Please let me know if this time works for you.

Best regards,
John Doe
Project Manager"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_complete_email_processing_workflow(self, mock_processor_class, mock_llm_responses, sample_email):
        """Test complete workflow from email loading to draft saving"""
        # Setup mock processor
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        # Create agent
        agent = ConversationalEmailAgent()
        
        # Step 1: Load email
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = json.loads(mock_llm_responses['extract_info'])
        mock_processor.text = sample_email
        
        response1 = agent.process_user_input(f"Here's an email I need help with: {sample_email}")
        
        assert "processed" in response1.lower() or "loaded" in response1.lower()
        assert agent.state_manager.context.current_state == ConversationState.EMAIL_LOADED
        mock_processor.load_text.assert_called_once()
        mock_processor.extract_key_info.assert_called_once()
        
        # Step 2: Draft reply
        mock_processor.draft_reply = Mock(return_value=mock_llm_responses['draft_reply'])
        mock_processor.last_draft = mock_llm_responses['draft_reply']
        
        response2 = agent.process_user_input("Draft a professional reply")
        
        assert "draft" in response2.lower()
        assert mock_llm_responses['draft_reply'] in response2
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
        mock_processor.draft_reply.assert_called_once()
        
        # Step 3: Refine draft
        mock_processor.refine = Mock(return_value=mock_llm_responses['refined_reply'])
        mock_processor.last_draft = mock_llm_responses['refined_reply']
        
        response3 = agent.process_user_input("Make it more formal")
        
        assert "refined" in response3.lower() or "updated" in response3.lower()
        assert mock_llm_responses['refined_reply'] in response3
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_REFINED
        mock_processor.refine.assert_called_once_with("Make it more formal")
        
        # Step 4: Save draft
        mock_processor.save_draft = Mock()
        
        response4 = agent.process_user_input("Save the draft to my_reply.txt")
        
        assert "saved" in response4.lower()
        assert agent.state_manager.context.current_state == ConversationState.READY_TO_SAVE
        mock_processor.save_draft.assert_called_once()
        
        # Verify conversation metrics
        assert agent.conversation_count == 4
        assert agent.successful_operations == 4
        assert agent.failed_operations == 0
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_error_recovery_workflow(self, mock_processor_class, sample_email):
        """Test error recovery in workflow"""
        # Setup mock processor
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Step 1: Successful email loading
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.text = sample_email
        
        response1 = agent.process_user_input(f"Process this email: {sample_email}")
        assert agent.state_manager.context.current_state == ConversationState.EMAIL_LOADED
        
        # Step 2: Failed draft creation
        mock_processor.draft_reply = Mock(side_effect=Exception("LLM service unavailable"))
        
        response2 = agent.process_user_input("Draft a reply")
        
        assert agent.state_manager.context.current_state == ConversationState.ERROR_RECOVERY
        assert agent.failed_operations == 1
        assert "error" in response2.lower() or "problem" in response2.lower()
        
        # Step 3: Recovery - try again
        mock_processor.draft_reply = Mock(return_value="Recovery draft")
        mock_processor.last_draft = "Recovery draft"
        
        response3 = agent.process_user_input("Try drafting again")
        
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
        assert agent.successful_operations == 2  # Load + successful draft
        assert "draft" in response3.lower()
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_multiple_email_processing(self, mock_processor_class, mock_llm_responses):
        """Test processing multiple emails in sequence"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Process first email
        email1 = "First email content"
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'First email summary'}
        mock_processor.text = email1
        
        response1 = agent.process_user_input(f"Process this email: {email1}")
        assert agent.state_manager.context.current_state == ConversationState.EMAIL_LOADED
        
        # Process second email (should reset context)
        email2 = "Second email content"
        mock_processor.key_info = {'summary': 'Second email summary'}
        mock_processor.text = email2
        
        response2 = agent.process_user_input(f"Now process this email: {email2}")
        assert agent.state_manager.context.current_state == ConversationState.EMAIL_LOADED
        
        # Verify both emails were processed
        assert mock_processor.load_text.call_count == 2
        assert mock_processor.extract_key_info.call_count == 2


class TestCLIIntegration:
    """Test CLI integration with the assistant"""
    
    @pytest.fixture
    def runner(self):
        """CLI test runner"""
        return CliRunner()
    
    @pytest.fixture
    def mock_session(self):
        """Mock session for CLI tests"""
        with patch('src.cli.cli.session') as mock:
            mock.text = None
            mock.key_info = None
            mock.last_draft = None
            yield mock
    
    def test_cli_complete_workflow(self, runner, mock_session):
        """Test complete workflow through CLI"""
        # Step 1: Load email
        mock_session.load_text.return_value = None
        mock_session.extract_key_info.return_value = None
        mock_session.text = "email content"
        mock_session.key_info = {"summary": "test"}
        
        result1 = runner.invoke(cli, ['load', 'test email content'])
        assert result1.exit_code == 0
        assert "Email content loaded successfully" in result1.output
        
        # Step 2: Draft reply
        mock_session.draft_reply.return_value = "Draft reply content"
        mock_session.last_draft = "Draft reply content"
        
        result2 = runner.invoke(cli, ['draft', 'formal'])
        assert result2.exit_code == 0
        assert "Drafted reply:" in result2.output
        assert "Draft reply content" in result2.output
        
        # Step 3: Save draft
        mock_session.save_draft.return_value = None
        
        result3 = runner.invoke(cli, ['save', 'test_draft.txt'])
        assert result3.exit_code == 0
        assert "Draft saved successfully" in result3.output
        
        # Verify method calls
        mock_session.load_text.assert_called_once()
        mock_session.extract_key_info.assert_called_once()
        mock_session.draft_reply.assert_called_once_with(tone="formal")
        mock_session.save_draft.assert_called_once_with("test_draft.txt", cloud=False)
    
    def test_cli_error_handling(self, runner, mock_session):
        """Test CLI error handling"""
        # Test load error
        mock_session.load_text.side_effect = Exception("Load failed")
        
        result1 = runner.invoke(cli, ['load', 'bad email'])
        assert result1.exit_code == 0  # CLI doesn't exit with error code
        assert "⚠️ Error loading email: Load failed" in result1.output
        
        # Test draft error
        mock_session.text = "email"
        mock_session.key_info = "info"
        mock_session.draft_reply.side_effect = Exception("Draft failed")
        
        result2 = runner.invoke(cli, ['draft'])
        assert result2.exit_code == 0
        assert "⚠️ Error drafting reply: Draft failed" in result2.output
    
    def test_cli_cloud_saving(self, runner, mock_session):
        """Test CLI cloud saving functionality"""
        mock_session.last_draft = "Cloud draft"
        mock_session.save_draft.return_value = None
        
        result = runner.invoke(cli, ['save', '--cloud'])
        assert result.exit_code == 0
        assert "Draft saved" in result.output
        
        mock_session.save_draft.assert_called_once_with(None, cloud=True)


class TestFileProcessingIntegration:
    """Test file processing integration"""
    
    def test_pdf_file_processing_workflow(self):
        """Test processing PDF files end-to-end"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Mock PDF processing
            with patch('src.assistant.utils.extract_text_from_pdf') as mock_pdf_extract, \
                 patch('src.assistant.conversational_agent.EmailLLMProcessor') as mock_processor_class:
                
                mock_pdf_extract.return_value = "PDF email content"
                
                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_processor.text = None
                mock_processor.key_info = None
                mock_processor.last_draft = None
                mock_processor.history = []
                
                agent = ConversationalEmailAgent()
                
                # Setup mocks
                mock_processor.load_text = Mock()
                mock_processor.extract_key_info = Mock()
                mock_processor.key_info = {'summary': 'PDF email summary'}
                mock_processor.text = "PDF email content"
                
                response = agent.process_user_input(f"Process this PDF file: {tmp_path}")
                
                assert "processed" in response.lower() or "loaded" in response.lower()
                mock_processor.load_text.assert_called_once_with(f"Process this PDF file: {tmp_path}")
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_text_file_processing_workflow(self):
        """Test processing text files end-to-end"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("Text file email content\nFrom: test@example.com\nTo: user@example.com")
            tmp_path = tmp_file.name
        
        try:
            with patch('src.assistant.conversational_agent.EmailLLMProcessor') as mock_processor_class:
                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_processor.text = None
                mock_processor.key_info = None
                mock_processor.last_draft = None
                mock_processor.history = []
                
                agent = ConversationalEmailAgent()
                
                # Setup mocks
                mock_processor.load_text = Mock()
                mock_processor.extract_key_info = Mock()
                mock_processor.key_info = {'summary': 'Text file email summary'}
                mock_processor.text = "Text file email content\nFrom: test@example.com\nTo: user@example.com"
                
                response = agent.process_user_input(f"Load email from {tmp_path}")
                
                assert "processed" in response.lower() or "loaded" in response.lower()
                mock_processor.load_text.assert_called_once()
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestCloudIntegration:
    """Test cloud storage integration"""
    
    @patch('src.assistant.utils.boto3.client')
    def test_s3_saving_integration(self, mock_boto_client):
        """Test S3 saving integration"""
        # Mock S3 client
        mock_s3 = Mock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.return_value = None
        mock_s3.put_object.return_value = None
        
        with patch('src.assistant.conversational_agent.EmailLLMProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.text = "email content"
            mock_processor.key_info = {'summary': 'test'}
            mock_processor.last_draft = "Draft to save to cloud"
            mock_processor.history = []
            
            # Mock save_draft to call the real S3 saving logic
            def mock_save_draft(filepath=None, cloud=False):
                if cloud:
                    from src.assistant.utils import save_draft_to_s3
                    save_draft_to_s3(mock_processor.last_draft, "email-assistant-drafts", filepath)
            
            mock_processor.save_draft = mock_save_draft
            
            agent = ConversationalEmailAgent()
            
            response = agent.process_user_input("Save this draft to cloud storage")
            
            assert "saved" in response.lower()
            mock_s3.put_object.assert_called_once()
            
            # Verify S3 call parameters
            call_args = mock_s3.put_object.call_args
            assert call_args[1]['Bucket'] == 'email-assistant-drafts'
            assert call_args[1]['Body'] == b'Draft to save to cloud'


class TestConversationFlow:
    """Test conversation flow and state management"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_conversation_context_preservation(self, mock_processor_class):
        """Test that conversation context is preserved across interactions"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Build up conversation context
        responses = []
        
        # Load email
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Meeting request'}
        mock_processor.text = "email content"
        
        responses.append(agent.process_user_input("Here's an email about a meeting"))
        
        # Draft reply
        mock_processor.draft_reply = Mock(return_value="Meeting reply draft")
        mock_processor.last_draft = "Meeting reply draft"
        
        responses.append(agent.process_user_input("Draft a reply"))
        
        # Refine with context
        mock_processor.refine = Mock(return_value="Refined meeting reply")
        mock_processor.last_draft = "Refined meeting reply"
        
        responses.append(agent.process_user_input("Make it more professional"))
        
        # Verify conversation history is maintained
        history = agent.state_manager.context.conversation_history
        assert len(history) == 6  # 3 user + 3 assistant messages
        
        # Verify context contains all information
        assert agent.state_manager.context.email_content is not None
        assert agent.state_manager.context.extracted_info is not None
        assert agent.state_manager.context.current_draft is not None
        assert len(agent.state_manager.context.draft_history) > 0
        
        # Verify state progression
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_REFINED
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_conversation_reset_and_new_session(self, mock_processor_class):
        """Test conversation reset functionality"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Process first email
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'First email'}
        mock_processor.text = "first email"
        
        agent.process_user_input("Process first email")
        
        # Verify state
        assert agent.conversation_count == 1
        assert agent.state_manager.context.current_state == ConversationState.EMAIL_LOADED
        
        # Reset conversation
        agent.reset_conversation()
        
        # Verify reset
        assert agent.conversation_count == 0
        assert agent.successful_operations == 0
        assert agent.failed_operations == 0
        assert agent.state_manager.context.current_state == ConversationState.GREETING
        
        # Process new email after reset
        mock_processor.key_info = {'summary': 'Second email'}
        mock_processor.text = "second email"
        
        agent.process_user_input("Process second email")
        
        assert agent.conversation_count == 1
        assert agent.state_manager.context.current_state == ConversationState.EMAIL_LOADED


@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance aspects of the integration"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_large_email_processing(self, mock_processor_class):
        """Test processing very large emails"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        # Create large email content
        large_email = "Large email content. " * 10000  # ~200KB of text
        
        agent = ConversationalEmailAgent()
        
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Large email summary'}
        mock_processor.text = large_email
        
        # Should handle large emails without issues
        response = agent.process_user_input(f"Process this large email: {large_email}")
        
        assert "processed" in response.lower() or "loaded" in response.lower()
        mock_processor.load_text.assert_called_once()
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_many_conversation_turns(self, mock_processor_class):
        """Test handling many conversation turns"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "draft"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Simulate many conversation turns
        for i in range(50):
            mock_processor.refine = Mock(return_value=f"Refined draft {i}")
            mock_processor.last_draft = f"Refined draft {i}"
            
            response = agent.process_user_input(f"Refine iteration {i}")
            assert isinstance(response, str)
            assert len(response) > 0
        
        # Verify conversation metrics
        assert agent.conversation_count == 50
        assert len(agent.state_manager.context.conversation_history) == 100  # 50 user + 50 assistant