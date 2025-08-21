"""
Comprehensive unit tests for conversation state management.
"""

import pytest
from datetime import datetime
from unittest.mock import patch

from assistant.conversation_state import (
    ConversationState,
    ConversationContext,
    ConversationStateManager
)


class TestConversationContext:
    """Test the ConversationContext dataclass"""
    
    def test_default_initialization(self):
        """Test that context initializes with correct defaults"""
        context = ConversationContext()
        
        assert context.current_state == ConversationState.GREETING
        assert context.email_content is None
        assert context.extracted_info is None
        assert context.current_draft is None
        assert context.draft_history == []
        assert context.user_preferences == {}
        assert context.conversation_history == []
        assert context.last_intent is None
        assert context.pending_clarification is None
        assert isinstance(context.session_start_time, datetime)
    
    def test_add_to_history(self):
        """Test adding messages to conversation history"""
        context = ConversationContext()
        
        with patch('assistant.conversation_state.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
            
            context.add_to_history("user", "Hello")
            context.add_to_history("assistant", "Hi there!")
            
            assert len(context.conversation_history) == 2
            assert context.conversation_history[0]["role"] == "user"
            assert context.conversation_history[0]["content"] == "Hello"
            assert context.conversation_history[0]["timestamp"] == "2023-01-01T12:00:00"
            assert context.conversation_history[1]["role"] == "assistant"
            assert context.conversation_history[1]["content"] == "Hi there!"
    
    def test_get_recent_history(self):
        """Test retrieving recent conversation history"""
        context = ConversationContext()
        
        # Add more messages than the limit
        for i in range(10):
            context.add_to_history("user", f"Message {i}")
        
        recent = context.get_recent_history(3)
        assert len(recent) == 3
        assert recent[0]["content"] == "Message 7"
        assert recent[1]["content"] == "Message 8"
        assert recent[2]["content"] == "Message 9"
    
    def test_get_recent_history_fewer_than_limit(self):
        """Test getting recent history when fewer messages exist"""
        context = ConversationContext()
        context.add_to_history("user", "Only message")
        
        recent = context.get_recent_history(5)
        assert len(recent) == 1
        assert recent[0]["content"] == "Only message"
    
    def test_reset_email_context(self):
        """Test resetting email-specific context"""
        context = ConversationContext()
        
        # Set up some email context
        context.email_content = "Test email"
        context.extracted_info = {"sender": "test@example.com"}
        context.current_draft = "Test draft"
        context.draft_history = ["Draft 1", "Draft 2"]
        context.current_state = ConversationState.DRAFT_CREATED
        
        # Reset email context
        context.reset_email_context()
        
        assert context.email_content is None
        assert context.extracted_info is None
        assert context.current_draft is None
        assert context.draft_history == []
        assert context.current_state == ConversationState.WAITING_FOR_EMAIL


class TestConversationStateManager:
    """Test the ConversationStateManager class"""
    
    def test_initialization(self):
        """Test state manager initialization"""
        manager = ConversationStateManager()
        
        assert isinstance(manager.context, ConversationContext)
        assert manager.context.current_state == ConversationState.GREETING
        assert hasattr(manager, 'transitions')
        assert isinstance(manager.transitions, dict)
    
    def test_valid_state_transitions(self):
        """Test valid state transitions"""
        manager = ConversationStateManager()
        
        # Test GREETING -> EMAIL_LOADED
        new_state = manager.transition_state('LOAD_EMAIL', success=True)
        assert new_state == ConversationState.EMAIL_LOADED
        assert manager.context.current_state == ConversationState.EMAIL_LOADED
        assert manager.context.last_intent == 'LOAD_EMAIL'
        
        # Test EMAIL_LOADED -> INFO_EXTRACTED
        new_state = manager.transition_state('EXTRACT_INFO', success=True)
        assert new_state == ConversationState.INFO_EXTRACTED
        assert manager.context.current_state == ConversationState.INFO_EXTRACTED
    
    def test_invalid_state_transition(self):
        """Test invalid state transitions stay in current state"""
        manager = ConversationStateManager()
        initial_state = manager.context.current_state
        
        # Try invalid transition from GREETING
        new_state = manager.transition_state('SAVE_DRAFT', success=True)
        
        # Should stay in same state
        assert new_state == initial_state
        assert manager.context.current_state == initial_state
    
    def test_failed_operation_goes_to_error_recovery(self):
        """Test that failed operations transition to error recovery"""
        manager = ConversationStateManager()
        manager.context.current_state = ConversationState.EMAIL_LOADED
        
        new_state = manager.transition_state('DRAFT_REPLY', success=False)
        
        assert new_state == ConversationState.ERROR_RECOVERY
        assert manager.context.current_state == ConversationState.ERROR_RECOVERY
    
    def test_can_transition(self):
        """Test checking if transitions are valid"""
        manager = ConversationStateManager()
        
        # From GREETING state
        assert manager.can_transition('LOAD_EMAIL') is True
        assert manager.can_transition('GENERAL_HELP') is True
        assert manager.can_transition('SAVE_DRAFT') is False
        
        # Change state and test again
        manager.context.current_state = ConversationState.DRAFT_CREATED
        assert manager.can_transition('SAVE_DRAFT') is True
        assert manager.can_transition('REFINE_DRAFT') is True
        assert manager.can_transition('LOAD_EMAIL') is True  # Can always load new email
    
    def test_get_valid_intents(self):
        """Test getting valid intents for current state"""
        manager = ConversationStateManager()
        
        # From GREETING state
        valid_intents = manager.get_valid_intents()
        expected_intents = ['LOAD_EMAIL', 'EXTRACT_INFO', 'DRAFT_REPLY', 'GENERAL_HELP', 'CLARIFICATION_NEEDED', 'VIEW_SESSION_HISTORY', 'VIEW_SPECIFIC_SESSION']
        assert set(valid_intents) == set(expected_intents)
        
        # From DRAFT_CREATED state
        manager.context.current_state = ConversationState.DRAFT_CREATED
        valid_intents = manager.get_valid_intents()
        expected_intents = ['REFINE_DRAFT', 'SAVE_DRAFT', 'CONTINUE_WORKFLOW', 'DECLINE_OFFER', 'DRAFT_REPLY', 'EXTRACT_INFO', 'LOAD_EMAIL', 'VIEW_SESSION_HISTORY', 'VIEW_SPECIFIC_SESSION']
        assert set(valid_intents) == set(expected_intents)
    
    def test_update_context(self):
        """Test updating context with new information"""
        manager = ConversationStateManager()
        
        manager.update_context(
            email_content="Test email",
            extracted_info={"sender": "test@example.com"},
            current_draft="Test draft"
        )
        
        assert manager.context.email_content == "Test email"
        assert manager.context.extracted_info == {"sender": "test@example.com"}
        assert manager.context.current_draft == "Test draft"
    
    def test_update_context_invalid_attribute(self):
        """Test updating context with invalid attributes (should be ignored)"""
        manager = ConversationStateManager()
        
        # This should not raise an error, just ignore invalid attributes
        manager.update_context(
            email_content="Test email",
            invalid_attribute="Should be ignored"
        )
        
        assert manager.context.email_content == "Test email"
        assert not hasattr(manager.context, 'invalid_attribute')
    
    def test_get_context_summary(self):
        """Test getting context summary"""
        manager = ConversationStateManager()
        
        # Set up some context
        manager.context.current_state = ConversationState.DRAFT_CREATED
        manager.context.email_content = "Test email"
        manager.context.extracted_info = {"sender": "test@example.com"}
        manager.context.current_draft = "Test draft"
        manager.context.draft_history = ["Draft 1", "Draft 2"]
        manager.context.conversation_history = [{"role": "user", "content": "Hello"}]
        manager.context.last_intent = "DRAFT_REPLY"
        
        summary = manager.get_context_summary()
        
        assert summary["current_state"] == "draft_created"
        assert summary["has_email"] is True
        assert summary["has_extracted_info"] is True
        assert summary["has_draft"] is True
        assert summary["draft_count"] == 2
        assert summary["conversation_length"] == 1
        assert summary["last_intent"] == "DRAFT_REPLY"
    
    def test_complex_workflow_transitions(self):
        """Test a complete workflow through multiple state transitions"""
        manager = ConversationStateManager()
        
        # Start at GREETING
        assert manager.context.current_state == ConversationState.GREETING
        
        # Load email
        manager.transition_state('LOAD_EMAIL', success=True)
        assert manager.context.current_state == ConversationState.EMAIL_LOADED
        
        # Extract info
        manager.transition_state('EXTRACT_INFO', success=True)
        assert manager.context.current_state == ConversationState.INFO_EXTRACTED
        
        # Draft reply
        manager.transition_state('DRAFT_REPLY', success=True)
        assert manager.context.current_state == ConversationState.DRAFT_CREATED
        
        # Refine draft
        manager.transition_state('REFINE_DRAFT', success=True)
        assert manager.context.current_state == ConversationState.DRAFT_REFINED
        
        # Save draft
        manager.transition_state('SAVE_DRAFT', success=True)
        assert manager.context.current_state == ConversationState.READY_TO_SAVE
        
        # Complete conversation
        manager.transition_state('SAVE_DRAFT', success=True)
        assert manager.context.current_state == ConversationState.CONVERSATION_COMPLETE
    
    def test_error_recovery_transitions(self):
        """Test transitions from error recovery state"""
        manager = ConversationStateManager()
        manager.context.current_state = ConversationState.ERROR_RECOVERY
        
        # Can load new email from error state
        manager.transition_state('LOAD_EMAIL', success=True)
        assert manager.context.current_state == ConversationState.EMAIL_LOADED
        
        # Reset to error recovery
        manager.context.current_state = ConversationState.ERROR_RECOVERY
        
        # Can save draft from error state
        manager.transition_state('SAVE_DRAFT', success=True)
        assert manager.context.current_state == ConversationState.CONVERSATION_COMPLETE
        
        # Reset to error recovery
        manager.context.current_state = ConversationState.ERROR_RECOVERY
        
        # Can extract info from error state
        manager.transition_state('EXTRACT_INFO', success=True)
        assert manager.context.current_state == ConversationState.INFO_EXTRACTED
        
        # Reset to error recovery
        manager.context.current_state = ConversationState.ERROR_RECOVERY
        
        # Can refine draft from error state
        manager.transition_state('REFINE_DRAFT', success=True)
        assert manager.context.current_state == ConversationState.DRAFT_REFINED
        
        # Reset to error recovery
        manager.context.current_state = ConversationState.ERROR_RECOVERY
        
        # Can get help from error state
        manager.transition_state('GENERAL_HELP', success=True)
        assert manager.context.current_state == ConversationState.GREETING
    
    def test_multiple_email_processing(self):
        """Test processing multiple emails in sequence"""
        manager = ConversationStateManager()
        
        # Process first email
        manager.transition_state('LOAD_EMAIL', success=True)
        manager.transition_state('DRAFT_REPLY', success=True)
        manager.transition_state('SAVE_DRAFT', success=True)
        assert manager.context.current_state == ConversationState.READY_TO_SAVE
        
        # Load new email (should work from any state)
        manager.transition_state('LOAD_EMAIL', success=True)
        assert manager.context.current_state == ConversationState.EMAIL_LOADED
    
    def test_ready_to_save_extract_info_transition(self):
        """Test that EXTRACT_INFO transition works from READY_TO_SAVE state"""
        manager = ConversationStateManager()
        
        # Set to READY_TO_SAVE state
        manager.context.current_state = ConversationState.READY_TO_SAVE
        
        # Should be able to transition to INFO_EXTRACTED with EXTRACT_INFO intent
        assert manager.can_transition('EXTRACT_INFO') is True
        
        new_state = manager.transition_state('EXTRACT_INFO', success=True)
        assert new_state == ConversationState.INFO_EXTRACTED
        assert manager.context.current_state == ConversationState.INFO_EXTRACTED


@pytest.mark.parametrize("initial_state,intent,expected_state", [
    (ConversationState.GREETING, 'LOAD_EMAIL', ConversationState.EMAIL_LOADED),
    (ConversationState.EMAIL_LOADED, 'CONTINUE_WORKFLOW', ConversationState.INFO_EXTRACTED),
    (ConversationState.INFO_EXTRACTED, 'CONTINUE_WORKFLOW', ConversationState.DRAFT_CREATED),
    (ConversationState.DRAFT_CREATED, 'CONTINUE_WORKFLOW', ConversationState.READY_TO_SAVE),
    (ConversationState.DRAFT_REFINED, 'SAVE_DRAFT', ConversationState.READY_TO_SAVE),
])
def test_state_transition_matrix(initial_state, intent, expected_state):
    """Test specific state transitions using parametrized tests"""
    manager = ConversationStateManager()
    manager.context.current_state = initial_state
    
    new_state = manager.transition_state(intent, success=True)
    assert new_state == expected_state
    assert manager.context.current_state == expected_state