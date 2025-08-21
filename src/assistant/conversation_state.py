"""
Conversation state management for the conversational email agent.
Tracks conversation flow, context, and state transitions.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ConversationState(Enum):
    """Possible states in the email processing conversation flow"""
    GREETING = "greeting"
    WAITING_FOR_EMAIL = "waiting_for_email"
    EMAIL_LOADED = "email_loaded"
    INFO_EXTRACTED = "info_extracted"
    DRAFT_CREATED = "draft_created"
    DRAFT_REFINED = "draft_refined"
    READY_TO_SAVE = "ready_to_save"
    CONVERSATION_COMPLETE = "conversation_complete"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class ConversationContext:
    """Maintains the context and state of the conversation"""
    current_state: ConversationState = ConversationState.GREETING
    email_content: Optional[str] = None
    extracted_info: Optional[Dict[str, Any]] = None
    current_draft: Optional[str] = None
    draft_history: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    last_intent: Optional[str] = None
    pending_clarification: Optional[str] = None
    session_start_time: datetime = field(default_factory=datetime.now)
    
    def add_to_history(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_recent_history(self, limit: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]
    
    def reset_email_context(self):
        """Reset email-specific context for processing a new email"""
        self.email_content = None
        self.extracted_info = None
        self.current_draft = None
        self.draft_history = []
        self.current_state = ConversationState.WAITING_FOR_EMAIL


class ConversationStateManager:
    """Manages conversation state transitions and context"""
    
    def __init__(self):
        self.context = ConversationContext()
        self._setup_state_transitions()
    
    def _setup_state_transitions(self):
        """Define valid state transitions based on intents"""
        self.transitions = {
            ConversationState.GREETING: {
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,
                'EXTRACT_INFO': ConversationState.INFO_EXTRACTED,  # Allow direct transition for auto-extraction
                'GENERAL_HELP': ConversationState.GREETING,
                'CLARIFICATION_NEEDED': ConversationState.GREETING,
            },
            ConversationState.WAITING_FOR_EMAIL: {
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,
                'GENERAL_HELP': ConversationState.WAITING_FOR_EMAIL,
            },
            ConversationState.EMAIL_LOADED: {
                'EXTRACT_INFO': ConversationState.INFO_EXTRACTED,
                'DRAFT_REPLY': ConversationState.DRAFT_CREATED,
                'CONTINUE_WORKFLOW': ConversationState.INFO_EXTRACTED,
                'DECLINE_OFFER': ConversationState.EMAIL_LOADED,  # Stay in same state
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,  # New email
            },
            ConversationState.INFO_EXTRACTED: {
                'DRAFT_REPLY': ConversationState.DRAFT_CREATED,
                'CONTINUE_WORKFLOW': ConversationState.DRAFT_CREATED,
                'DECLINE_OFFER': ConversationState.INFO_EXTRACTED,  # Stay in same state
                'EXTRACT_INFO': ConversationState.INFO_EXTRACTED,  # Allow re-showing info
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,  # New email
                'CLARIFICATION_NEEDED': ConversationState.INFO_EXTRACTED,  # Handle clarification requests
            },
            ConversationState.DRAFT_CREATED: {
                'REFINE_DRAFT': ConversationState.DRAFT_REFINED,
                'SAVE_DRAFT': ConversationState.READY_TO_SAVE,
                'CONTINUE_WORKFLOW': ConversationState.READY_TO_SAVE,
                'DECLINE_OFFER': ConversationState.DRAFT_CREATED,  # Stay in same state
                'DRAFT_REPLY': ConversationState.DRAFT_CREATED,  # New draft
                'EXTRACT_INFO': ConversationState.DRAFT_CREATED,  # Allow showing info without changing state
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,  # New email
            },
            ConversationState.DRAFT_REFINED: {
                'REFINE_DRAFT': ConversationState.DRAFT_REFINED,  # Multiple refinements
                'SAVE_DRAFT': ConversationState.READY_TO_SAVE,
                'CONTINUE_WORKFLOW': ConversationState.READY_TO_SAVE,
                'DECLINE_OFFER': ConversationState.DRAFT_REFINED,  # Stay in same state
                'DRAFT_REPLY': ConversationState.DRAFT_CREATED,  # Start over
                'EXTRACT_INFO': ConversationState.DRAFT_REFINED,  # Allow showing info without changing state
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,  # New email
            },
            ConversationState.READY_TO_SAVE: {
                'SAVE_DRAFT': ConversationState.CONVERSATION_COMPLETE,
                'REFINE_DRAFT': ConversationState.DRAFT_REFINED,  # More changes
                'DRAFT_REPLY': ConversationState.DRAFT_CREATED,  # New draft
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,  # New email
            },
            ConversationState.CONVERSATION_COMPLETE: {
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,  # New email
                'GENERAL_HELP': ConversationState.GREETING,
            },
            ConversationState.ERROR_RECOVERY: {
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,
                'DRAFT_REPLY': ConversationState.DRAFT_CREATED,
                'SAVE_DRAFT': ConversationState.CONVERSATION_COMPLETE,  # Allow saving from error recovery
                'EXTRACT_INFO': ConversationState.INFO_EXTRACTED,  # Allow showing info from error recovery
                'REFINE_DRAFT': ConversationState.DRAFT_REFINED,  # Allow refining from error recovery
                'GENERAL_HELP': ConversationState.GREETING,
                'CLARIFICATION_NEEDED': ConversationState.ERROR_RECOVERY,
            }
        }
    
    def transition_state(self, intent: str, success: bool = True) -> ConversationState:
        """
        Transition to the next state based on current state, intent, and success
        
        Args:
            intent: The classified intent from user input
            success: Whether the operation was successful
            
        Returns:
            The new conversation state
        """
        if not success:
            self.context.current_state = ConversationState.ERROR_RECOVERY
            return self.context.current_state
        
        current_state = self.context.current_state
        valid_transitions = self.transitions.get(current_state, {})
        
        if intent in valid_transitions:
            new_state = valid_transitions[intent]
            self.context.current_state = new_state
            self.context.last_intent = intent
        else:
            # Invalid transition - stay in current state
            print(f"Warning: Invalid transition from {current_state} with intent {intent}")
        
        return self.context.current_state
    
    def can_transition(self, intent: str) -> bool:
        """Check if a transition is valid from the current state"""
        current_state = self.context.current_state
        return intent in self.transitions.get(current_state, {})
    
    def get_valid_intents(self) -> List[str]:
        """Get list of valid intents from current state"""
        current_state = self.context.current_state
        return list(self.transitions.get(current_state, {}).keys())
    
    def update_context(self, **kwargs):
        """Update context with new information"""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context for debugging or logging"""
        return {
            "current_state": self.context.current_state.value,
            "has_email": self.context.email_content is not None,
            "has_extracted_info": self.context.extracted_info is not None,
            "has_draft": self.context.current_draft is not None,
            "draft_count": len(self.context.draft_history),
            "conversation_length": len(self.context.conversation_history),
            "last_intent": self.context.last_intent,
        }