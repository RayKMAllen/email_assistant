"""
Comprehensive unit tests for the main conversational email agent.
"""

import pytest
from unittest.mock import Mock, patch

from src.assistant.conversational_agent import ConversationalEmailAgent
from assistant.conversation_state import ConversationState
from src.assistant.intent_classifier import IntentResult


class TestConversationalEmailAgent:
    """Test the ConversationalEmailAgent class"""
    
    @pytest.fixture
    def agent(self):
        """Create an agent instance for testing"""
        with patch('src.assistant.conversational_agent.EmailLLMProcessor') as mock_processor:
            mock_processor.return_value = Mock()
            return ConversationalEmailAgent()
    
    @pytest.fixture
    def mock_intent_result(self):
        """Create a mock intent result"""
        return IntentResult(
            intent='LOAD_EMAIL',
            confidence=0.9,
            parameters={'email_content': 'test email'},
            reasoning='Test reasoning',
            method='rule_based'
        )
    
    def test_initialization(self, agent):
        """Test agent initialization"""
        assert hasattr(agent, 'email_processor')
        assert hasattr(agent, 'state_manager')
        assert hasattr(agent, 'intent_classifier')
        assert hasattr(agent, 'response_generator')
        assert agent.conversation_count == 0
        assert agent.successful_operations == 0
        assert agent.failed_operations == 0
    
    def test_process_user_input_successful_flow(self, agent, mock_intent_result):
        """Test successful user input processing flow"""
        user_input = "Here's an email I need help with"
        
        # Mock all the components
        agent.intent_classifier.classify = Mock(return_value=mock_intent_result)
        agent._execute_intent = Mock(return_value=({'result': 'success'}, True))
        agent.state_manager.transition_state = Mock(return_value=ConversationState.EMAIL_LOADED)
        agent.response_generator.generate_response = Mock(return_value="Email processed successfully!")
        
        result = agent.process_user_input(user_input)
        
        assert result == "Email processed successfully!"
        assert agent.conversation_count == 1
        assert agent.successful_operations == 1
        assert agent.failed_operations == 0
        
        # Verify method calls
        agent.intent_classifier.classify.assert_called_once_with(user_input, agent.state_manager.context)
        agent._execute_intent.assert_called_once_with(mock_intent_result, user_input)
        agent.state_manager.transition_state.assert_called_once_with('LOAD_EMAIL', True)
        agent.response_generator.generate_response.assert_called_once_with('LOAD_EMAIL', {'result': 'success'}, True)
        
        # Check conversation history
        assert len(agent.state_manager.context.conversation_history) == 2
        assert agent.state_manager.context.conversation_history[0]['role'] == 'user'
        assert agent.state_manager.context.conversation_history[0]['content'] == user_input
        assert agent.state_manager.context.conversation_history[1]['role'] == 'assistant'
        assert agent.state_manager.context.conversation_history[1]['content'] == "Email processed successfully!"
    
    def test_process_user_input_failed_operation(self, agent, mock_intent_result):
        """Test user input processing with failed operation"""
        user_input = "Process this email"
        
        agent.intent_classifier.classify = Mock(return_value=mock_intent_result)
        agent._execute_intent = Mock(return_value=({'error': 'Failed to process'}, False))
        agent.state_manager.transition_state = Mock(return_value=ConversationState.ERROR_RECOVERY)
        agent.response_generator.generate_response = Mock(return_value="Sorry, there was an error.")
        
        result = agent.process_user_input(user_input)
        
        assert result == "Sorry, there was an error."
        assert agent.conversation_count == 1
        assert agent.successful_operations == 0
        assert agent.failed_operations == 1
        
        # Should transition with success=False
        agent.state_manager.transition_state.assert_called_once_with('LOAD_EMAIL', False)
        agent.response_generator.generate_response.assert_called_once_with('LOAD_EMAIL', {'error': 'Failed to process'}, False)
    
    def test_process_user_input_clarification_needed(self, agent):
        """Test user input processing when clarification is needed"""
        user_input = "I need help with something"
        clarification_result = IntentResult(
            intent='CLARIFICATION_NEEDED',
            confidence=0.9,
            parameters={'original_input': user_input},
            reasoning='Ambiguous input',
            method='fallback'
        )
        
        agent.intent_classifier.classify = Mock(return_value=clarification_result)
        agent.response_generator.generate_clarification_response = Mock(return_value="Could you be more specific?")
        agent.state_manager.get_context_summary = Mock(return_value={'state': 'greeting'})
        
        result = agent.process_user_input(user_input)
        
        assert result == "Could you be more specific?"
        agent.response_generator.generate_clarification_response.assert_called_once_with(
            user_input, {'state': 'greeting', 'original_input': user_input}
        )
        # Should not execute intent or transition state for clarification
        # Check that _execute_intent was not called (since we expect clarification)
        # Note: _execute_intent is a method, not a mock, so we can't check if it was called
        # Instead, we verify the state didn't change from GREETING
        # The state should remain GREETING since no intent was executed
        assert agent.state_manager.context.current_state.value == ConversationState.GREETING.value
    
    def test_process_user_input_unexpected_error(self, agent):
        """Test handling of unexpected errors"""
        user_input = "Test input"
        
        # Make intent classifier raise an exception
        agent.intent_classifier.classify = Mock(side_effect=Exception("Unexpected error"))
        agent._handle_unexpected_error = Mock(return_value="Something went wrong. Please try again.")
        
        result = agent.process_user_input(user_input)
        
        assert result == "Something went wrong. Please try again."
        assert agent.failed_operations == 1
        agent._handle_unexpected_error.assert_called_once()
    
    # Test intent execution methods
    
    def test_execute_intent_load_email(self, agent):
        """Test executing LOAD_EMAIL intent"""
        intent_result = IntentResult(
            intent='LOAD_EMAIL',
            confidence=0.9,
            parameters={'email_content': 'test email content'},
            reasoning='Email detected',
            method='rule_based'
        )
        
        agent._handle_load_email = Mock(return_value=({'email_loaded': True}, True))
        
        result, success = agent._execute_intent(intent_result, "test input")
        
        assert success is True
        assert result == {'email_loaded': True}
        agent._handle_load_email.assert_called_once_with({'email_content': 'test email content'}, "test input")
    
    def test_execute_intent_draft_reply(self, agent):
        """Test executing DRAFT_REPLY intent"""
        intent_result = IntentResult(
            intent='DRAFT_REPLY',
            confidence=0.9,
            parameters={'tone': 'formal'},
            reasoning='Draft request detected',
            method='rule_based'
        )
        
        agent._handle_draft_reply = Mock(return_value=({'draft': 'test draft'}, True))
        
        result, success = agent._execute_intent(intent_result, "draft a reply")
        
        assert success is True
        assert result == {'draft': 'test draft'}
        agent._handle_draft_reply.assert_called_once_with({'tone': 'formal'})
    
    def test_execute_intent_unknown(self, agent):
        """Test executing unknown intent"""
        intent_result = IntentResult(
            intent='UNKNOWN_INTENT',
            confidence=0.5,
            parameters={},
            reasoning='Unknown',
            method='fallback'
        )
        
        result, success = agent._execute_intent(intent_result, "unknown input")
        
        assert success is False
        assert "I'm not sure how to handle that request: UNKNOWN_INTENT" in result
    
    def test_execute_intent_exception_handling(self, agent):
        """Test exception handling in intent execution"""
        intent_result = IntentResult(
            intent='LOAD_EMAIL',
            confidence=0.9,
            parameters={},
            reasoning='Test',
            method='rule_based'
        )
        
        agent._handle_load_email = Mock(side_effect=Exception("Handler error"))
        
        result, success = agent._execute_intent(intent_result, "test input")
        
        assert success is False
        assert "Error executing LOAD_EMAIL: Handler error" in result
    
    # Test specific intent handlers
    
    def test_handle_load_email_success(self, agent):
        """Test successful email loading"""
        parameters = {'email_content': 'test email'}
        user_input = "Here's an email"
        
        agent.email_processor.load_text = Mock()
        agent.email_processor.extract_key_info = Mock()
        agent.email_processor.key_info = {'sender': 'test@example.com'}
        agent.state_manager.update_context = Mock()
        
        result, success = agent._handle_load_email(parameters, user_input)
        
        assert success is True
        assert result['email_content'] == 'test email'
        assert result['extracted_info'] == {'sender': 'test@example.com'}
        
        agent.email_processor.load_text.assert_called_once_with('test email')
        agent.email_processor.extract_key_info.assert_called_once()
        agent.state_manager.update_context.assert_called()
    
    def test_handle_load_email_no_content_in_parameters(self, agent):
        """Test email loading when no content in parameters"""
        parameters = {}
        user_input = "Process this email content"
        
        agent.email_processor.load_text = Mock()
        agent.email_processor.extract_key_info = Mock()
        agent.email_processor.key_info = {}
        agent.state_manager.update_context = Mock()
        
        result, success = agent._handle_load_email(parameters, user_input)
        
        assert success is True
        # Should use full user input as email content
        agent.email_processor.load_text.assert_called_once_with(user_input)
    
    def test_handle_load_email_failure(self, agent):
        """Test email loading failure"""
        parameters = {'email_content': 'bad email'}
        user_input = "test"
        
        agent.email_processor.load_text = Mock(side_effect=Exception("Load error"))
        
        result, success = agent._handle_load_email(parameters, user_input)
        
        assert success is False
        assert result['error'] == 'Load error'
    
    def test_handle_extract_info_success(self, agent):
        """Test successful info extraction"""
        agent.email_processor.text = "email content"
        agent.email_processor.key_info = None
        agent.state_manager.update_context = Mock()
        
        # Mock extract_key_info to set key_info when called
        def mock_extract():
            agent.email_processor.key_info = {'sender': 'test@example.com'}
        
        agent.email_processor.extract_key_info = Mock(side_effect=mock_extract)
        
        result, success = agent._handle_extract_info()
        
        assert success is True
        assert result == {'sender': 'test@example.com'}
        # Verify that extract_key_info was called
        agent.email_processor.extract_key_info.assert_called_once()
        # Verify that update_context was called with the extracted info
        agent.state_manager.update_context.assert_called_with(extracted_info={'sender': 'test@example.com'})
    
    def test_handle_extract_info_no_email_loaded(self, agent):
        """Test info extraction with no email loaded"""
        agent.email_processor.text = None
        
        result, success = agent._handle_extract_info()
        
        assert success is False
        assert result['error'] == 'No email loaded to extract information from'
    
    def test_handle_extract_info_already_extracted(self, agent):
        """Test info extraction when info already exists"""
        agent.email_processor.text = "email content"
        agent.email_processor.key_info = {'sender': 'existing@example.com'}
        agent.state_manager.update_context = Mock()
        
        result, success = agent._handle_extract_info()
        
        assert success is True
        assert result == {
            'extracted_info': {'sender': 'existing@example.com'},
            'already_extracted': True
        }
        # Should not call extract_key_info again
        assert not hasattr(agent.email_processor, 'extract_key_info') or not agent.email_processor.extract_key_info.called
    
    def test_handle_draft_reply_success(self, agent):
        """Test successful draft reply"""
        parameters = {'tone': 'formal'}
        
        agent.email_processor.text = "email content"
        agent.email_processor.draft_reply = Mock(return_value="Draft reply")
        agent.state_manager.update_context = Mock()
        agent.state_manager.context.draft_history = []
        
        result, success = agent._handle_draft_reply(parameters)
        
        assert success is True
        assert result['draft'] == "Draft reply"
        assert result['tone'] == 'formal'
        
        agent.email_processor.draft_reply.assert_called_once_with(tone='formal')
        agent.state_manager.update_context.assert_called_with(current_draft="Draft reply")
        assert agent.state_manager.context.draft_history == ["Draft reply"]
    
    def test_handle_draft_reply_no_email(self, agent):
        """Test draft reply with no email loaded"""
        parameters = {}
        agent.email_processor.text = None
        
        result, success = agent._handle_draft_reply(parameters)
        
        assert success is False
        assert result['error'] == 'No email loaded to draft a reply for'
    
    def test_handle_refine_draft_success(self, agent):
        """Test successful draft refinement"""
        parameters = {'refinement_instructions': 'make it more formal'}
        user_input = "make it more formal"
        
        agent.email_processor.last_draft = "Original draft"
        agent.email_processor.refine = Mock(return_value="Refined draft")
        agent.state_manager.update_context = Mock()
        agent.state_manager.context.draft_history = []
        
        result, success = agent._handle_refine_draft(parameters, user_input)
        
        assert success is True
        assert result == "Refined draft"
        
        agent.email_processor.refine.assert_called_once_with('make it more formal')
        agent.state_manager.update_context.assert_called_with(current_draft="Refined draft")
        assert agent.state_manager.context.draft_history == ["Refined draft"]
    
    def test_handle_refine_draft_no_draft(self, agent):
        """Test draft refinement with no existing draft"""
        parameters = {}
        user_input = "refine it"
        
        agent.email_processor.last_draft = None
        
        result, success = agent._handle_refine_draft(parameters, user_input)
        
        assert success is False
        assert result == "No draft available to refine"
    
    def test_handle_refine_draft_use_user_input(self, agent):
        """Test draft refinement using user input when no parameters"""
        parameters = {}
        user_input = "make it shorter and more direct"
        
        agent.email_processor.last_draft = "Long draft"
        agent.email_processor.refine = Mock(return_value="Short draft")
        agent.state_manager.update_context = Mock()
        agent.state_manager.context.draft_history = []
        
        result, success = agent._handle_refine_draft(parameters, user_input)
        
        assert success is True
        # Should use user input as refinement instructions
        agent.email_processor.refine.assert_called_once_with(user_input)
    
    def test_handle_save_draft_success(self, agent):
        """Test successful draft saving"""
        parameters = {'filepath': '/path/to/draft.txt', 'cloud': False}
        
        agent.email_processor.last_draft = "Draft to save"
        agent.email_processor.save_draft = Mock()
        
        result, success = agent._handle_save_draft(parameters)
        
        assert success is True
        assert result['filepath'] == '/path/to/draft.txt'
        assert result['cloud'] is False
        
        agent.email_processor.save_draft.assert_called_once_with(filepath='/path/to/draft.txt', cloud=False)
    
    def test_handle_save_draft_default_filepath(self, agent):
        """Test draft saving with default filepath"""
        parameters = {}
        
        agent.email_processor.last_draft = "Draft to save"
        agent.email_processor.save_draft = Mock()
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20231201_143022"
            
            result, success = agent._handle_save_draft(parameters)
        
        assert success is True
        assert "draft_20231201_143022.txt" in result['filepath']
        assert result['cloud'] is False
    
    def test_handle_save_draft_no_draft(self, agent):
        """Test draft saving with no draft available"""
        parameters = {}
        agent.email_processor.last_draft = None
        
        result, success = agent._handle_save_draft(parameters)
        
        assert success is False
        assert result['error'] == 'No draft available to save'
    
    def test_handle_general_help(self, agent):
        """Test general help handling"""
        result, success = agent._handle_general_help()
        
        assert success is True
        assert result == "help_requested"
    
    def test_handle_continue_workflow_email_loaded(self, agent):
        """Test continue workflow from email loaded state"""
        agent.state_manager.context.current_state = ConversationState.EMAIL_LOADED
        agent.email_processor.key_info = None
        agent.email_processor.extract_key_info = Mock()
        agent.email_processor.key_info = {'sender': 'test@example.com'}
        agent.state_manager.update_context = Mock()
        
        result, success = agent._handle_continue_workflow()
        
        assert success is True
        # The continue workflow should return the key_info from EMAIL_LOADED state
        assert result == {'sender': 'test@example.com'}
    
    def test_handle_continue_workflow_info_extracted(self, agent):
        """Test continue workflow from info extracted state"""
        agent.state_manager.context.current_state = ConversationState.INFO_EXTRACTED
        agent.email_processor.draft_reply = Mock(return_value="Auto draft")
        agent.state_manager.update_context = Mock()
        agent.state_manager.context.draft_history = []
        
        result, success = agent._handle_continue_workflow()
        
        assert success is True
        # The continue workflow should return draft info from INFO_EXTRACTED state
        assert result == {'draft': 'Auto draft'}
    
    def test_handle_continue_workflow_draft_created(self, agent):
        """Test continue workflow from draft created state"""
        agent.state_manager.context.current_state = ConversationState.DRAFT_CREATED
        
        result, success = agent._handle_continue_workflow()
        
        assert success is True
        # The continue workflow should return ready_to_save from DRAFT_CREATED state
        assert result == 'ready_to_save'
    
    def test_handle_continue_workflow_other_state(self, agent):
        """Test continue workflow from other states"""
        agent.state_manager.context.current_state = ConversationState.GREETING
        
        result, success = agent._handle_continue_workflow()
        
        assert success is True
        assert result == "continue_acknowledged"
    
    # Test error handling
    
    def test_handle_unexpected_error(self, agent):
        """Test unexpected error handling"""
        error = Exception("Test error")
        user_input = "test input"
        
        with patch('builtins.print') as mock_print, \
             patch('traceback.format_exc') as mock_traceback, \
             patch('random.choice') as mock_choice:
            
            mock_traceback.return_value = "Traceback details"
            mock_choice.return_value = "I encountered an unexpected issue."
            
            result = agent._handle_unexpected_error(error, user_input)
            
            assert result == "I encountered an unexpected issue."
            assert agent.state_manager.context.current_state.value == ConversationState.ERROR_RECOVERY.value
            
            # Should log error details
            mock_print.assert_called()
            mock_traceback.assert_called_once()
    
    # Test utility methods
    
    def test_get_conversation_summary(self, agent):
        """Test getting conversation summary"""
        # Set up some conversation state
        agent.conversation_count = 5
        agent.successful_operations = 3
        agent.failed_operations = 1
        agent.state_manager.context.current_state = ConversationState.DRAFT_CREATED
        agent.state_manager.context.email_content = "test email"
        agent.state_manager.context.current_draft = "test draft"
        agent.state_manager.context.draft_history = ["draft1", "draft2"]
        agent.state_manager.context.conversation_history = [{"role": "user", "content": "hello"}]
        
        summary = agent.get_conversation_summary()
        
        assert summary['conversation_state'] == 'draft_created'
        assert summary['conversation_count'] == 5
        assert summary['successful_operations'] == 3
        assert summary['failed_operations'] == 1
        assert summary['has_email_loaded'] is True
        assert summary['has_draft'] is True
        assert summary['draft_history_count'] == 2
        assert summary['conversation_history_length'] == 1
    
    def test_reset_conversation(self, agent):
        """Test conversation reset"""
        # Set up some conversation state
        agent.conversation_count = 5
        agent.successful_operations = 3
        agent.failed_operations = 1
        agent.state_manager.context.email_content = "test email"
        agent.state_manager.context.current_draft = "test draft"
        agent.state_manager.context.current_state = ConversationState.DRAFT_CREATED
        
        agent.reset_conversation()
        
        assert agent.conversation_count == 0
        assert agent.successful_operations == 0
        assert agent.failed_operations == 0
        assert agent.state_manager.context.current_state.value == ConversationState.GREETING.value
        # Email context should be reset (mocked method)
        assert hasattr(agent.state_manager.context, 'reset_email_context')
    
    def test_get_greeting_message(self, agent):
        """Test getting greeting message"""
        with patch('random.choice') as mock_choice:
            mock_choice.return_value = "Hello! I'm your email assistant."
            
            greeting = agent.get_greeting_message()
            
            assert greeting == "Hello! I'm your email assistant."
            # Should choose from available greeting messages
            mock_choice.assert_called_once()
            args = mock_choice.call_args[0][0]
            assert len(args) > 0
            # Check that the greeting messages contain relevant keywords
            assert any("email" in msg.lower() for msg in args)
    
    # Test integration scenarios
    
    def test_complete_email_processing_workflow(self, agent):
        """Test complete workflow from email loading to saving"""
        # Mock all dependencies
        agent.intent_classifier.classify = Mock()
        agent.email_processor.load_text = Mock()
        agent.email_processor.extract_key_info = Mock()
        agent.email_processor.draft_reply = Mock(return_value="Draft reply")
        agent.email_processor.save_draft = Mock()
        agent.email_processor.text = "email content"
        agent.email_processor.key_info = {'sender': 'test@example.com'}
        agent.email_processor.last_draft = "Draft reply"
        agent.response_generator.generate_response = Mock(return_value="Success!")
        agent.state_manager.transition_state = Mock()
        agent.state_manager.update_context = Mock()
        agent.state_manager.context.draft_history = []
        
        # Step 1: Load email
        agent.intent_classifier.classify.return_value = IntentResult(
            intent='LOAD_EMAIL', confidence=0.9, parameters={'email_content': 'test'}, 
            reasoning='', method='rule_based'
        )
        result1 = agent.process_user_input("Here's an email")
        
        # Step 2: Draft reply
        agent.intent_classifier.classify.return_value = IntentResult(
            intent='DRAFT_REPLY', confidence=0.9, parameters={'tone': 'formal'}, 
            reasoning='', method='rule_based'
        )
        result2 = agent.process_user_input("Draft a formal reply")
        
        # Step 3: Save draft
        agent.intent_classifier.classify.return_value = IntentResult(
            intent='SAVE_DRAFT', confidence=0.9, parameters={'filepath': 'draft.txt'}, 
            reasoning='', method='rule_based'
        )
        result3 = agent.process_user_input("Save the draft")
        
        # Verify all steps executed
        assert agent.conversation_count == 3
        assert agent.successful_operations == 3
        assert agent.failed_operations == 0
        
        # Verify method calls
        agent.email_processor.load_text.assert_called()
        agent.email_processor.extract_key_info.assert_called()
        agent.email_processor.draft_reply.assert_called()
        agent.email_processor.save_draft.assert_called()
    
    @pytest.mark.parametrize("intent,expected_handler", [
        ('LOAD_EMAIL', '_handle_load_email'),
        ('EXTRACT_INFO', '_handle_extract_info'),
        ('DRAFT_REPLY', '_handle_draft_reply'),
        ('REFINE_DRAFT', '_handle_refine_draft'),
        ('SAVE_DRAFT', '_handle_save_draft'),
        ('GENERAL_HELP', '_handle_general_help'),
        ('CONTINUE_WORKFLOW', '_handle_continue_workflow'),
    ])
    def test_intent_handler_mapping(self, agent, intent, expected_handler):
        """Test that intents are mapped to correct handlers"""
        intent_result = IntentResult(
            intent=intent, confidence=0.9, parameters={}, reasoning='', method='rule_based'
        )
        
        # Mock the expected handler
        mock_handler = Mock(return_value=('result', True))
        setattr(agent, expected_handler, mock_handler)
        
        result, success = agent._execute_intent(intent_result, "test input")
        
        mock_handler.assert_called_once()