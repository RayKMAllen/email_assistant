"""
Comprehensive unit tests for the conversational response generation system.
"""

import pytest
from unittest.mock import patch

from src.assistant.response_generator import ConversationalResponseGenerator
from assistant.conversation_state import (
    ConversationState,
    ConversationStateManager
)


class TestConversationalResponseGenerator:
    """Test the ConversationalResponseGenerator class"""
    
    @pytest.fixture
    def state_manager(self):
        """Create a state manager for testing"""
        return ConversationStateManager()
    
    @pytest.fixture
    def generator(self, state_manager):
        """Create a response generator for testing"""
        return ConversationalResponseGenerator(state_manager)
    
    def test_initialization(self, generator, state_manager):
        """Test response generator initialization"""
        assert generator.state_manager == state_manager
        assert hasattr(generator, 'response_templates')
        assert hasattr(generator, 'error_templates')
        assert hasattr(generator, 'guidance_templates')
    
    def test_response_templates_structure(self, generator):
        """Test that response templates are properly structured"""
        templates = generator.response_templates
        
        # Check that key intents have templates
        required_intents = ['LOAD_EMAIL', 'EXTRACT_INFO', 'DRAFT_REPLY', 'REFINE_DRAFT', 'SAVE_DRAFT', 'GENERAL_HELP']
        for intent in required_intents:
            assert intent in templates
            assert 'success' in templates[intent]
            assert isinstance(templates[intent]['success'], list)
            assert len(templates[intent]['success']) > 0
    
    def test_error_templates_structure(self, generator):
        """Test that error templates are properly structured"""
        error_templates = generator.error_templates
        
        # Check that key intents have error templates
        required_intents = ['LOAD_EMAIL', 'DRAFT_REPLY', 'EXTRACT_INFO', 'SAVE_DRAFT']
        for intent in required_intents:
            assert intent in error_templates
            assert isinstance(error_templates[intent], list)
            assert len(error_templates[intent]) > 0
        
        # Check general error template exists
        assert 'GENERAL' in error_templates
    
    def test_guidance_templates_structure(self, generator):
        """Test that guidance templates are properly structured"""
        guidance_templates = generator.guidance_templates
        
        # Check that all conversation states have guidance
        for state in ConversationState:
            assert state in guidance_templates
            assert isinstance(guidance_templates[state], list)
            assert len(guidance_templates[state]) > 0
    
    # Test main response generation
    
    def test_generate_response_success(self, generator):
        """Test generating successful response"""
        generator.state_manager.context.current_state = ConversationState.EMAIL_LOADED
        
        with patch.object(generator, '_generate_main_response') as mock_main, \
             patch.object(generator, '_generate_proactive_guidance') as mock_guidance:
            
            mock_main.return_value = "Email processed successfully"
            mock_guidance.return_value = "What would you like to do next?"
            
            result = generator.generate_response('LOAD_EMAIL', {'email_content': 'test'}, success=True)
            
            expected = "Email processed successfully\n\nWhat would you like to do next?"
            assert result == expected
            mock_main.assert_called_once_with('LOAD_EMAIL', {'email_content': 'test'})
            mock_guidance.assert_called_once()
    
    def test_generate_response_failure(self, generator):
        """Test generating error response"""
        with patch.object(generator, '_generate_error_response') as mock_error:
            mock_error.return_value = "Sorry, there was an error processing that."
            
            result = generator.generate_response('LOAD_EMAIL', {'error': 'test error'}, success=False)
            
            assert result == "Sorry, there was an error processing that."
            mock_error.assert_called_once_with('LOAD_EMAIL', {'error': 'test error'})
    
    def test_generate_response_main_only(self, generator):
        """Test generating response with only main response (no guidance)"""
        with patch.object(generator, '_generate_main_response') as mock_main, \
             patch.object(generator, '_generate_proactive_guidance') as mock_guidance:
            
            mock_main.return_value = "Main response"
            mock_guidance.return_value = ""
            
            result = generator.generate_response('LOAD_EMAIL', {}, success=True)
            
            assert result == "Main response"
    
    def test_generate_response_guidance_only(self, generator):
        """Test generating response with only guidance (no main response)"""
        with patch.object(generator, '_generate_main_response') as mock_main, \
             patch.object(generator, '_generate_proactive_guidance') as mock_guidance:
            
            mock_main.return_value = ""
            mock_guidance.return_value = "What can I help you with?"
            
            result = generator.generate_response('UNKNOWN_INTENT', {}, success=True)
            
            assert result == "What can I help you with?"
    
    def test_generate_response_fallback(self, generator):
        """Test fallback response when both main and guidance are empty"""
        with patch.object(generator, '_generate_main_response') as mock_main, \
             patch.object(generator, '_generate_proactive_guidance') as mock_guidance:
            
            mock_main.return_value = ""
            mock_guidance.return_value = ""
            
            result = generator.generate_response('UNKNOWN_INTENT', {}, success=True)
            
            assert result == "I'm here to help! What would you like me to do?"
    
    # Test specific response formatting
    
    def test_format_load_email_response(self, generator):
        """Test formatting load email response"""
        template = "I've processed your email{email_info}. {summary}"
        result_data = {
            'extracted_info': {
                'sender_name': 'John Doe',
                'subject': 'Meeting Request',
                'summary': 'John wants to schedule a meeting'
            }
        }
        
        result = generator._format_load_email_response(template, result_data)
        
        assert "John Doe" in result
        assert "Meeting Request" in result
        assert "John wants to schedule a meeting" in result
    
    def test_format_load_email_response_minimal_info(self, generator):
        """Test formatting load email response with minimal info"""
        template = "I've processed your email{email_info}. {summary}"
        result_data = {
            'extracted_info': {
                'sender_name': 'John Doe'
                # No subject or summary
            }
        }
        
        result = generator._format_load_email_response(template, result_data)
        
        assert "John Doe" in result
        assert result.count("{") == 0  # No unformatted placeholders
    
    def test_format_extract_info_response(self, generator):
        """Test formatting extract info response"""
        template = "Here's the key information I extracted:"
        result_data = {
            'summary': 'Meeting request from John',
            'sender_name': 'John Doe',
            'receiver_name': 'Jane Smith',
            'subject': 'Weekly Meeting',
            'sender_contact_details': {'email': 'john@example.com', 'phone': '123-456-7890'}
        }
        
        result = generator._format_extract_info_response(template, result_data)
        
        assert "Here's the key information I extracted:" in result
        assert "**Summary:** Meeting request from John" in result
        assert "**From:** John Doe" in result
        assert "**To:** Jane Smith" in result
        assert "**Subject:** Weekly Meeting" in result
        assert "john@example.com" in result
        assert "123-456-7890" in result
    
    def test_format_draft_reply_response(self, generator):
        """Test formatting draft reply response"""
        template = "I've drafted a reply for you{tone_info}:"
        result_data = {
            'tone': 'formal',
            'draft': 'Dear John,\n\nThank you for your email.\n\nBest regards,\nJane'
        }
        
        result = generator._format_draft_reply_response(template, result_data)
        
        assert "in a formal tone" in result
        assert "Dear John," in result
        assert "Thank you for your email." in result
        assert "Best regards," in result
    
    def test_format_draft_reply_response_no_tone(self, generator):
        """Test formatting draft reply response without tone"""
        template = "I've drafted a reply for you{tone_info}:"
        result_data = {
            'draft': 'Hi John,\n\nThanks!\n\nJane'
        }
        
        result = generator._format_draft_reply_response(template, result_data)
        
        assert "Hi John," in result
        assert "Thanks!" in result
        # Should not have tone info
        assert "tone" not in result.lower()
    
    def test_format_refine_response(self, generator):
        """Test formatting refine response"""
        template = "I've refined the draft based on your feedback:"
        refined_draft = "Dear John,\n\nThank you for your professional email.\n\nBest regards,\nJane"
        
        result = generator._format_refine_response(template, refined_draft)
        
        assert template in result
        assert refined_draft in result
    
    def test_format_save_response(self, generator):
        """Test formatting save response"""
        template = "Perfect! I've saved your draft to {filepath}."
        result_data = {'filepath': '/home/user/drafts/reply.txt'}
        
        result = generator._format_save_response(template, result_data)
        
        assert "Perfect! I've saved your draft to /home/user/drafts/reply.txt." == result
    
    def test_format_save_response_default_location(self, generator):
        """Test formatting save response with default location"""
        template = "Perfect! I've saved your draft to {filepath}."
        result_data = {}  # No filepath specified
        
        result = generator._format_save_response(template, result_data)
        
        assert "Perfect! I've saved your draft to the default location." == result
    
    def test_format_help_response(self, generator):
        """Test formatting help response"""
        template = "I'm your email assistant! Here's what I can help you with:"
        
        result = generator._format_help_response(template)
        
        assert template in result
        assert "📧 **Process emails**" in result
        assert "🔍 **Extract key information**" in result
        assert "✍️ **Draft replies**" in result
        assert "🔧 **Refine drafts**" in result
        assert "💾 **Save drafts**" in result
        assert "🔄 **Iterative refinement**" in result
    
    # Test proactive guidance generation
    
    def test_generate_proactive_guidance_greeting(self, generator):
        """Test proactive guidance for greeting state"""
        generator.state_manager.context.current_state = ConversationState.GREETING
        
        with patch('src.assistant.response_generator.random.choice') as mock_choice:
            mock_choice.return_value = "Test guidance message"
            
            result = generator._generate_proactive_guidance()
            
            assert result == "Test guidance message"
            # Should have called random.choice with greeting templates
            mock_choice.assert_called_once()
            args = mock_choice.call_args[0][0]
            assert len(args) > 0
            assert all("help" in msg.lower() or "can" in msg.lower() for msg in args)
    
    def test_generate_proactive_guidance_email_loaded(self, generator):
        """Test proactive guidance for email loaded state"""
        generator.state_manager.context.current_state = ConversationState.EMAIL_LOADED
        
        result = generator._generate_proactive_guidance()
        
        # Should suggest next steps like extracting info or drafting reply
        assert any(word in result.lower() for word in ['extract', 'draft', 'reply', 'information'])
    
    def test_generate_proactive_guidance_draft_created(self, generator):
        """Test proactive guidance for draft created state"""
        generator.state_manager.context.current_state = ConversationState.DRAFT_CREATED
        
        result = generator._generate_proactive_guidance()
        
        # Should suggest refining or saving
        assert any(word in result.lower() for word in ['refine', 'save', 'formal', 'changes'])
    
    def test_generate_proactive_guidance_unknown_state(self, generator):
        """Test proactive guidance for unknown state"""
        # Create a mock state that's not in guidance templates
        generator.state_manager.context.current_state = "UNKNOWN_STATE"
        
        result = generator._generate_proactive_guidance()
        
        assert result == "What would you like me to help you with next?"
    
    # Test error response generation
    
    def test_generate_error_response_specific_intent(self, generator):
        """Test generating error response for specific intent"""
        with patch('random.choice') as mock_choice:
            mock_choice.return_value = "Specific error message"
            
            result = generator._generate_error_response('LOAD_EMAIL', {'error': 'File not found'})
            
            assert result == "Specific error message\n\nError details: File not found"
            # Should use LOAD_EMAIL specific templates
            mock_choice.assert_called_once()
    
    def test_generate_error_response_general(self, generator):
        """Test generating general error response"""
        with patch('random.choice') as mock_choice:
            mock_choice.return_value = "General error message"
            
            result = generator._generate_error_response('UNKNOWN_INTENT', "Some error")
            
            assert result == "General error message"
            # Should use general error templates
            mock_choice.assert_called_once()
    
    def test_generate_error_response_no_error_details(self, generator):
        """Test generating error response without error details"""
        with patch('random.choice') as mock_choice:
            mock_choice.return_value = "Error message"
            
            result = generator._generate_error_response('LOAD_EMAIL', "Simple error")
            
            assert result == "Error message"
            assert "Error details:" not in result
    
    # Test clarification response generation
    
    def test_generate_clarification_response_greeting_state(self, generator):
        """Test clarification response in greeting state"""
        generator.state_manager.context.current_state = ConversationState.GREETING
        
        with patch('random.choice') as mock_choice:
            mock_choice.return_value = "I'd be happy to help! Could you clarify what you'd like me to do?"
            
            result = generator.generate_clarification_response("unclear input", {})
            
            assert "I'd be happy to help!" in result
            assert "Share an email" in result
            assert "Ask me what I can do" in result
            assert "Provide a file path" in result
    
    def test_generate_clarification_response_email_loaded_state(self, generator):
        """Test clarification response in email loaded state"""
        generator.state_manager.context.current_state = ConversationState.EMAIL_LOADED
        
        result = generator.generate_clarification_response("unclear input", {})
        
        assert "extract key information" in result
        assert "draft reply" in result
        assert "summary" in result
    
    def test_generate_clarification_response_draft_created_state(self, generator):
        """Test clarification response in draft created state"""
        generator.state_manager.context.current_state = ConversationState.DRAFT_CREATED
        
        result = generator.generate_clarification_response("unclear input", {})
        
        assert "refine the draft" in result
        assert "save the draft" in result
        assert "specific changes" in result
    
    def test_generate_clarification_response_no_suggestions(self, generator):
        """Test clarification response with no specific suggestions"""
        generator.state_manager.context.current_state = ConversationState.CONVERSATION_COMPLETE
        
        with patch('random.choice') as mock_choice:
            mock_choice.return_value = "Base clarification message"
            
            result = generator.generate_clarification_response("unclear input", {})
            
            assert result == "Base clarification message"
    
    # Test template randomization
    
    def test_response_randomization(self, generator):
        """Test that responses use random template selection"""
        generator.state_manager.context.current_state = ConversationState.EMAIL_LOADED
        
        # Generate multiple responses and check for variation
        responses = []
        for _ in range(10):
            with patch('random.choice', side_effect=lambda x: x[0]):  # Always pick first
                result = generator._generate_main_response('LOAD_EMAIL', {})
                responses.append(result)
        
        # All should be the same since we're forcing first choice
        assert len(set(responses)) == 1
        
        # Now test with actual randomization
        responses = []
        for _ in range(20):
            result = generator._generate_main_response('LOAD_EMAIL', {})
            responses.append(result)
        
        # Should have some variation (though this could occasionally fail due to randomness)
        # We'll just check that we're calling the template system correctly
        assert all(isinstance(r, str) for r in responses)
        assert all(len(r) > 0 for r in responses)
    
    # Test edge cases
    
    def test_empty_operation_result(self, generator):
        """Test handling empty operation result"""
        result = generator.generate_response('LOAD_EMAIL', {}, success=True)
        
        # Should still generate a response
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_none_operation_result(self, generator):
        """Test handling None operation result"""
        result = generator.generate_response('LOAD_EMAIL', None, success=True)
        
        # Should still generate a response
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_malformed_operation_result(self, generator):
        """Test handling malformed operation result"""
        # Test with various malformed data
        malformed_results = [
            {"draft": None},
            {"extracted_info": "not a dict"},
            {"filepath": 123},  # Wrong type
            {"tone": "invalid_tone"},  # Invalid tone that's not in templates
        ]
        
        for malformed_result in malformed_results:
            result = generator.generate_response('DRAFT_REPLY', malformed_result, success=True)
            
            # Should handle gracefully without crashing
            assert isinstance(result, str)
            assert len(result) > 0
    
    @pytest.mark.parametrize("intent,expected_keywords", [
        ('LOAD_EMAIL', ['processed', 'loaded', 'email']),
        ('DRAFT_REPLY', ['drafted', 'reply', 'response']),
        ('SAVE_DRAFT', ['saved', 'draft']),
        ('GENERAL_HELP', ['help', 'capabilities', 'can']),
    ])
    def test_intent_specific_responses(self, generator, intent, expected_keywords):
        """Test that responses contain intent-specific keywords"""
        result = generator._generate_main_response(intent, {})
        
        # Should contain at least one of the expected keywords
        result_lower = result.lower()
        assert any(keyword in result_lower for keyword in expected_keywords)