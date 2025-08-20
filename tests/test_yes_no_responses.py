"""
Tests for yes/no response handling in the conversational email agent.
Verifies that the agent correctly interprets yes/no responses to offers.
"""

import pytest
from unittest.mock import Mock, patch

from assistant.conversational_agent import ConversationalEmailAgent
from assistant.conversation_state import ConversationState
from assistant.intent_classifier import HybridIntentClassifier, IntentResult


class TestYesNoResponses:
    """Test yes/no response handling in different conversation contexts"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.agent = ConversationalEmailAgent()
        
        # Mock the email processor to avoid actual LLM calls
        self.agent.email_processor = Mock()
        self.agent.email_processor.text = "Sample email content"
        self.agent.email_processor.key_info = {
            'sender_name': 'John Doe',
            'subject': 'Test Email',
            'summary': 'This is a test email'
        }
        self.agent.email_processor.last_draft = "Sample draft reply"
        
        # Mock methods
        self.agent.email_processor.load_text = Mock()
        self.agent.email_processor.extract_key_info = Mock()
        self.agent.email_processor.draft_reply = Mock(return_value="Sample draft reply")
        self.agent.email_processor.refine = Mock(return_value="Refined draft reply")
        self.agent.email_processor.save_draft = Mock()
    
    def test_yes_response_after_info_extracted_offer(self):
        """Test 'yes' response when agent offers to draft a reply"""
        # Set up state: info has been extracted
        self.agent.state_manager.context.current_state = ConversationState.INFO_EXTRACTED
        self.agent.state_manager.context.email_content = "Sample email"
        self.agent.state_manager.context.extracted_info = self.agent.email_processor.key_info
        
        # Process 'yes' response
        response = self.agent.process_user_input("yes")
        
        # Verify that a draft was created
        self.agent.email_processor.draft_reply.assert_called_once()
        assert self.agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
        assert "draft" in response.lower()
    
    def test_no_response_after_info_extracted_offer(self):
        """Test 'no' response when agent offers to draft a reply"""
        # Set up state: info has been extracted
        self.agent.state_manager.context.current_state = ConversationState.INFO_EXTRACTED
        self.agent.state_manager.context.email_content = "Sample email"
        self.agent.state_manager.context.extracted_info = self.agent.email_processor.key_info
        
        # Process 'no' response
        response = self.agent.process_user_input("no")
        
        # Verify that no draft was created and state remained the same
        self.agent.email_processor.draft_reply.assert_not_called()
        assert self.agent.state_manager.context.current_state == ConversationState.INFO_EXTRACTED
        assert "no problem" in response.lower() or "that's fine" in response.lower()
    
    def test_yes_response_after_draft_created_offer(self):
        """Test 'yes' response when agent offers to save a draft"""
        # Set up state: draft has been created
        self.agent.state_manager.context.current_state = ConversationState.DRAFT_CREATED
        self.agent.state_manager.context.current_draft = "Sample draft"
        
        # Process 'yes' response
        response = self.agent.process_user_input("yes")
        
        # Verify that the state transitioned to ready to save
        assert self.agent.state_manager.context.current_state == ConversationState.READY_TO_SAVE
    
    def test_no_response_after_draft_created_offer(self):
        """Test 'no' response when agent offers to save a draft"""
        # Set up state: draft has been created
        self.agent.state_manager.context.current_state = ConversationState.DRAFT_CREATED
        self.agent.state_manager.context.current_draft = "Sample draft"
        
        # Process 'no' response
        response = self.agent.process_user_input("no")
        
        # Verify that state remained the same and appropriate response was given
        assert self.agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
        assert "fine" in response.lower() or "problem" in response.lower()
    
    def test_various_yes_patterns(self):
        """Test various ways of saying 'yes'"""
        yes_patterns = ["yes", "ok", "okay", "sure", "please do", "go for it", "do it"]
        
        for pattern in yes_patterns:
            # Reset state
            self.agent.state_manager.context.current_state = ConversationState.INFO_EXTRACTED
            self.agent.state_manager.context.email_content = "Sample email"
            self.agent.state_manager.context.extracted_info = self.agent.email_processor.key_info
            
            # Reset mock
            self.agent.email_processor.draft_reply.reset_mock()
            
            # Process the pattern
            response = self.agent.process_user_input(pattern)
            
            # Verify that a draft was created
            self.agent.email_processor.draft_reply.assert_called_once()
            assert self.agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
    
    def test_various_no_patterns(self):
        """Test various ways of saying 'no'"""
        no_patterns = ["no", "nope", "not now", "skip", "no thanks", "pass"]
        
        for pattern in no_patterns:
            # Reset state
            self.agent.state_manager.context.current_state = ConversationState.INFO_EXTRACTED
            self.agent.state_manager.context.email_content = "Sample email"
            self.agent.state_manager.context.extracted_info = self.agent.email_processor.key_info
            
            # Reset mock
            self.agent.email_processor.draft_reply.reset_mock()
            
            # Process the pattern
            response = self.agent.process_user_input(pattern)
            
            # Verify that no draft was created
            self.agent.email_processor.draft_reply.assert_not_called()
            assert self.agent.state_manager.context.current_state == ConversationState.INFO_EXTRACTED
    
    def test_intent_classification_confidence(self):
        """Test that yes/no responses have high confidence in appropriate contexts"""
        classifier = HybridIntentClassifier()
        
        # Test yes response in INFO_EXTRACTED state
        self.agent.state_manager.context.current_state = ConversationState.INFO_EXTRACTED
        result = classifier.classify("yes", self.agent.state_manager.context)
        
        assert result.intent == 'CONTINUE_WORKFLOW'
        assert result.confidence >= 0.9  # Should have high confidence
        
        # Test no response in INFO_EXTRACTED state
        result = classifier.classify("no", self.agent.state_manager.context)
        
        assert result.intent == 'DECLINE_OFFER'
        assert result.confidence >= 0.9  # Should have high confidence
    
    def test_context_dependent_classification(self):
        """Test that yes/no responses are classified differently based on context"""
        classifier = HybridIntentClassifier()
        
        # Test in different states
        states_to_test = [
            ConversationState.EMAIL_LOADED,
            ConversationState.INFO_EXTRACTED,
            ConversationState.DRAFT_CREATED,
            ConversationState.DRAFT_REFINED
        ]
        
        for state in states_to_test:
            self.agent.state_manager.context.current_state = state
            
            # Test yes response
            yes_result = classifier.classify("yes", self.agent.state_manager.context)
            assert yes_result.intent == 'CONTINUE_WORKFLOW'
            assert yes_result.confidence >= 0.9
            
            # Test no response
            no_result = classifier.classify("no", self.agent.state_manager.context)
            assert no_result.intent == 'DECLINE_OFFER'
            assert no_result.confidence >= 0.9
    
    def test_conversation_flow_with_yes_no(self):
        """Test complete conversation flow with yes/no responses"""
        # Start with email loaded
        self.agent.state_manager.context.current_state = ConversationState.EMAIL_LOADED
        self.agent.state_manager.context.email_content = "Sample email"
        
        # Agent should offer to extract info or draft reply
        # User says "yes" to continue workflow
        response1 = self.agent.process_user_input("yes")
        assert self.agent.state_manager.context.current_state == ConversationState.INFO_EXTRACTED
        
        # Agent should now offer to draft a reply
        # User says "yes" again
        response2 = self.agent.process_user_input("yes")
        assert self.agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
        
        # Agent should offer to save or refine
        # User says "no" to decline saving
        response3 = self.agent.process_user_input("no")
        assert self.agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
        assert "fine" in response3.lower() or "problem" in response3.lower()
    
    def test_edge_cases(self):
        """Test edge cases for yes/no responses"""
        # Test yes/no in greeting state (should have lower confidence)
        self.agent.state_manager.context.current_state = ConversationState.GREETING
        
        classifier = HybridIntentClassifier()
        result = classifier.classify("yes", self.agent.state_manager.context)
        
        # In greeting state, yes should still be classified but with context consideration
        assert result.intent in ['CONTINUE_WORKFLOW', 'CLARIFICATION_NEEDED']
        
        # Test mixed case and with punctuation
        self.agent.state_manager.context.current_state = ConversationState.INFO_EXTRACTED
        
        responses_to_test = ["Yes!", "YES", "yes.", "No!", "NO", "no."]
        for response in responses_to_test:
            result = classifier.classify(response, self.agent.state_manager.context)
            assert result.intent in ['CONTINUE_WORKFLOW', 'DECLINE_OFFER']
            assert result.confidence >= 0.7  # Base confidence should be at least 0.7