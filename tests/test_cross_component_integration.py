"""
Cross-component integration tests.
Tests interactions between different components of the email assistant system.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import json
import tempfile
import os
from datetime import datetime

from src.assistant.conversational_agent import ConversationalEmailAgent
from src.assistant.conversation_state import ConversationState, ConversationStateManager
from src.assistant.intent_classifier import HybridIntentClassifier
from src.assistant.response_generator import ConversationalResponseGenerator
from src.cli.cli import cli
from click.testing import CliRunner


class TestAgentComponentIntegration:
    """Test integration between agent and its core components"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_agent_state_manager_integration(self, mock_processor_class):
        """Test integration between agent and state manager"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Verify initial state
        assert agent.state_manager.context.current_state == ConversationState.GREETING
        
        # Load email and verify state transition
        email = "From: test@example.com\nSubject: Test\nContent"
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Test email'}
        mock_processor.text = email
        
        response1 = agent.process_user_input(f"Process: {email}")
        
        # Verify state manager updated correctly
        assert agent.state_manager.context.current_state == ConversationState.INFO_EXTRACTED
        assert agent.state_manager.context.email_content == email
        assert agent.state_manager.context.extracted_info == {'summary': 'Test email'}
        
        # Draft reply and verify state transition
        mock_processor.draft_reply = Mock(return_value="Draft reply")
        mock_processor.last_draft = "Draft reply"
        
        response2 = agent.process_user_input("Draft a reply")
        
        # Verify state manager updated correctly
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
        assert agent.state_manager.context.current_draft == "Draft reply"
        assert len(agent.state_manager.context.draft_history) == 1
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_agent_intent_classifier_integration(self, mock_processor_class):
        """Test integration between agent and intent classifier"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test that intent classifier receives correct context
        with patch.object(agent.intent_classifier, 'classify') as mock_classify:
            mock_classify.return_value = Mock(
                intent='GENERAL_HELP',
                confidence=0.9,
                parameters={},
                reasoning='Help requested',
                method='rule_based'
            )
            
            agent.process_user_input("help me")
            
            # Verify intent classifier was called with correct parameters
            mock_classify.assert_called_once()
            call_args = mock_classify.call_args
            assert call_args[0][0] == "help me"  # user input
            assert call_args[0][1] == agent.state_manager.context  # context
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_agent_response_generator_integration(self, mock_processor_class):
        """Test integration between agent and response generator"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'Test email'}
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test that response generator receives correct parameters
        with patch.object(agent.response_generator, 'generate_response') as mock_generate:
            mock_generate.return_value = "Generated response"
            
            # Setup for successful operation
            mock_processor.draft_reply = Mock(return_value="Draft content")
            mock_processor.last_draft = "Draft content"
            
            agent.process_user_input("draft a reply")
            
            # Verify response generator was called correctly
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert call_args[0][0] == 'DRAFT_REPLY'  # intent
            assert 'draft' in call_args[0][1]  # operation result
            assert call_args[0][2] == True  # success
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_component_error_propagation(self, mock_processor_class):
        """Test error propagation between components"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test error in LLM processor propagates correctly
        mock_processor.load_text = Mock(side_effect=Exception("LLM error"))
        
        response = agent.process_user_input("Process email: test content")
        
        # Verify error was handled and state updated
        assert agent.state_manager.context.current_state == ConversationState.ERROR_RECOVERY
        assert agent.failed_operations > 0
        assert "error" in response.lower() or "problem" in response.lower()


class TestStateManagerComponentIntegration:
    """Test state manager integration with other components"""
    
    def test_state_manager_context_integration(self):
        """Test state manager context integration"""
        state_manager = ConversationStateManager()
        
        # Test context updates
        state_manager.update_context(
            email_content="Test email",
            extracted_info={'summary': 'Test'},
            current_draft="Test draft"
        )
        
        # Verify context was updated correctly
        assert state_manager.context.email_content == "Test email"
        assert state_manager.context.extracted_info == {'summary': 'Test'}
        assert state_manager.context.current_draft == "Test draft"
        
        # Test conversation history
        state_manager.context.add_to_history("user", "Hello")
        state_manager.context.add_to_history("assistant", "Hi there!")
        
        assert len(state_manager.context.conversation_history) == 2
        assert state_manager.context.conversation_history[0]['role'] == 'user'
        assert state_manager.context.conversation_history[1]['role'] == 'assistant'
    
    def test_state_transitions_integration(self):
        """Test state transition logic integration"""
        state_manager = ConversationStateManager()
        
        # Test successful transitions
        new_state = state_manager.transition_state('LOAD_EMAIL', success=True)
        assert new_state == ConversationState.EMAIL_LOADED
        
        new_state = state_manager.transition_state('EXTRACT_INFO', success=True)
        assert new_state == ConversationState.INFO_EXTRACTED
        
        new_state = state_manager.transition_state('DRAFT_REPLY', success=True)
        assert new_state == ConversationState.DRAFT_CREATED
        
        # Test failed transitions
        new_state = state_manager.transition_state('SAVE_DRAFT', success=False)
        assert new_state == ConversationState.ERROR_RECOVERY
    
    def test_session_management_integration(self):
        """Test session management integration"""
        state_manager = ConversationStateManager()
        
        # Setup initial session
        state_manager.update_context(
            email_content="First email",
            extracted_info={'summary': 'First email summary'},
            current_draft="First draft"
        )
        
        # Archive current session
        state_manager.context.archive_current_email_session()
        
        # Verify session was archived
        assert len(state_manager.context.archived_sessions) == 1
        archived = state_manager.context.archived_sessions[0]
        assert archived.email_content == "First email"
        assert archived.current_draft == "First draft"
        
        # Setup new session
        state_manager.update_context(
            email_content="Second email",
            extracted_info={'summary': 'Second email summary'}
        )
        
        # Verify new session is active
        assert state_manager.context.email_content == "Second email"
        assert state_manager.context.current_draft is None  # Reset for new session


class TestIntentClassifierComponentIntegration:
    """Test intent classifier integration with other components"""
    
    def test_intent_classifier_context_awareness(self):
        """Test intent classifier context awareness"""
        from src.assistant.conversation_state import ConversationContext
        
        classifier = HybridIntentClassifier()
        context = ConversationContext()
        
        # Test context-dependent classification
        context.current_state = ConversationState.EMAIL_LOADED
        
        # "yes" should be classified as CONTINUE_WORKFLOW in this context
        result = classifier.classify("yes", context)
        assert result.intent == 'CONTINUE_WORKFLOW'
        assert result.confidence > 0.8
        
        # Change context
        context.current_state = ConversationState.DRAFT_CREATED
        
        # "yes" should still be CONTINUE_WORKFLOW but with different implications
        result = classifier.classify("yes", context)
        assert result.intent == 'CONTINUE_WORKFLOW'
    
    @patch('src.assistant.llm_session.EmailLLMProcessor')
    def test_intent_classifier_llm_integration(self, mock_processor_class):
        """Test intent classifier integration with LLM processor"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        classifier = HybridIntentClassifier(email_processor=mock_processor)
        context = Mock()
        context.current_state = ConversationState.GREETING
        context.get_recent_history = Mock(return_value=[])
        
        # Mock LLM response
        mock_processor.send_prompt = Mock(return_value=json.dumps({
            'intent': 'LOAD_EMAIL',
            'confidence': 0.9,
            'parameters': {'email_content': 'test email'},
            'reasoning': 'User provided email content'
        }))
        
        # Test ambiguous input that requires LLM
        result = classifier.classify("here's something I need help with", context)
        
        # Should use LLM classification
        assert result.method == 'llm_based'
        assert result.intent == 'LOAD_EMAIL'
        assert result.confidence == 0.9


class TestResponseGeneratorComponentIntegration:
    """Test response generator integration with other components"""
    
    def test_response_generator_state_awareness(self):
        """Test response generator state awareness"""
        state_manager = ConversationStateManager()
        response_generator = ConversationalResponseGenerator(state_manager)
        
        # Test response generation in different states
        state_manager.context.current_state = ConversationState.EMAIL_LOADED
        response = response_generator._generate_proactive_guidance()
        assert any(keyword in response.lower() for keyword in ['extract', 'draft', 'information'])
        
        state_manager.context.current_state = ConversationState.DRAFT_CREATED
        response = response_generator._generate_proactive_guidance()
        assert any(keyword in response.lower() for keyword in ['refine', 'save', 'formal', 'changes'])
    
    def test_response_generator_context_formatting(self):
        """Test response generator context-aware formatting"""
        state_manager = ConversationStateManager()
        response_generator = ConversationalResponseGenerator(state_manager)
        
        # Test email loading response formatting
        result = {
            'email_content': 'test email',
            'extracted_info': {
                'sender_name': 'John Doe',
                'subject': 'Test Subject',
                'summary': 'This is a test email'
            },
            'auto_extracted': True
        }
        
        response = response_generator._format_load_email_response(
            "I've processed your email{email_info}. {summary}",
            result
        )
        
        assert 'John Doe' in response
        assert 'Test Subject' in response
        assert 'test email' in response


class TestCLIAgentIntegration:
    """Test CLI integration with agent components"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('src.cli.cli.get_agent')
    def test_cli_agent_lifecycle_integration(self, mock_get_agent, runner):
        """Test CLI agent lifecycle integration"""
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        
        # Test agent creation and reuse
        mock_agent.process_user_input.return_value = "Response 1"
        result1 = runner.invoke(cli, ['ask', 'message 1'])
        
        mock_agent.process_user_input.return_value = "Response 2"
        result2 = runner.invoke(cli, ['ask', 'message 2'])
        
        # Verify same agent instance was used
        assert mock_get_agent.call_count >= 2
        assert mock_agent.process_user_input.call_count == 2
    
    @patch('src.cli.cli.get_agent')
    def test_cli_agent_state_persistence(self, mock_get_agent, runner):
        """Test CLI agent state persistence across commands"""
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        
        # Setup mock responses
        mock_agent.get_conversation_summary.return_value = {
            'conversation_state': 'email_loaded',
            'conversation_count': 2,
            'successful_operations': 1,
            'failed_operations': 0,
            'has_email_loaded': True,
            'has_draft': False,
            'draft_history_count': 0
        }
        
        # Test status command
        result = runner.invoke(cli, ['status'])
        assert result.exit_code == 0
        assert 'email_loaded' in result.output
        assert '2' in result.output  # conversation count
        
        # Test reset command
        result = runner.invoke(cli, ['reset'])
        assert result.exit_code == 0
        mock_agent.reset_conversation.assert_called_once()


class TestFileSystemComponentIntegration:
    """Test file system integration with components"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_file_processing_component_integration(self, mock_processor_class):
        """Test file processing integration across components"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("From: test@example.com\nSubject: Test\nContent")
            tmp_path = tmp_file.name
        
        try:
            # Mock file processing
            mock_processor.load_text = Mock()
            mock_processor.extract_key_info = Mock()
            mock_processor.key_info = {'summary': 'File processed successfully'}
            mock_processor.text = "File content"
            
            response = agent.process_user_input(f"Process file: {tmp_path}")
            
            # Verify integration worked
            assert "processed" in response.lower() or "loaded" in response.lower()
            mock_processor.load_text.assert_called_once()
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_save_operation_component_integration(self, mock_processor_class):
        """Test save operation integration across components"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "Draft to save"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Setup successful save
        mock_processor.save_draft = Mock()
        
        response = agent.process_user_input("save the draft")
        
        # Verify save integration
        assert "saved" in response.lower()
        mock_processor.save_draft.assert_called_once()


class TestCloudIntegrationComponents:
    """Test cloud service integration with components"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    @patch('src.assistant.utils.boto3.client')
    def test_s3_integration_components(self, mock_boto_client, mock_processor_class):
        """Test S3 integration across components"""
        # Setup S3 mock
        mock_s3 = Mock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.return_value = None
        mock_s3.put_object.return_value = None
        
        # Setup processor mock
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "Draft for cloud"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Mock save_draft to use S3
        def mock_save_draft(filepath=None, cloud=False):
            if cloud:
                from src.assistant.utils import save_draft_to_s3
                save_draft_to_s3(mock_processor.last_draft, "test-bucket", filepath)
        
        mock_processor.save_draft = mock_save_draft
        
        response = agent.process_user_input("save to cloud storage")
        
        # Verify cloud integration
        assert "saved" in response.lower()
        mock_s3.put_object.assert_called_once()


class TestErrorHandlingComponentIntegration:
    """Test error handling integration across components"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_component_error_recovery_integration(self, mock_processor_class):
        """Test error recovery integration across components"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test error in one component affects others correctly
        mock_processor.draft_reply = Mock(side_effect=Exception("Component error"))
        
        response = agent.process_user_input("draft a reply")
        
        # Verify error handling integration
        assert agent.state_manager.context.current_state == ConversationState.ERROR_RECOVERY
        assert agent.failed_operations > 0
        assert "error" in response.lower() or "problem" in response.lower()
        
        # Test recovery
        mock_processor.draft_reply = Mock(return_value="Recovery draft")
        mock_processor.last_draft = "Recovery draft"
        
        response = agent.process_user_input("try again")
        
        # Verify recovery integration
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
        assert "draft" in response.lower()


class TestPerformanceComponentIntegration:
    """Test performance aspects of component integration"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_component_performance_integration(self, mock_processor_class):
        """Test performance integration across components"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "draft"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test performance with multiple component interactions
        import time
        start_time = time.time()
        
        for i in range(50):
            mock_processor.refine = Mock(return_value=f"Draft {i}")
            mock_processor.last_draft = f"Draft {i}"
            
            response = agent.process_user_input(f"refine {i}")
            assert isinstance(response, str)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete reasonably quickly (adjust threshold as needed)
        assert duration < 5.0  # 5 seconds for 50 operations
        
        # Verify components maintained consistency
        assert agent.conversation_count == 50
        assert len(agent.state_manager.context.conversation_history) == 100  # 50 user + 50 assistant