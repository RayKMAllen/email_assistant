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
class EmailSession:
    """Represents a complete email processing session"""
    email_content: str
    extracted_info: Optional[Dict[str, Any]] = None
    drafts: List[str] = field(default_factory=list)
    current_draft: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    email_id: Optional[str] = None  # For identification


@dataclass
class ConversationContext:
    """Maintains the context and state of the conversation"""
    current_state: ConversationState = ConversationState.GREETING
    
    # Current email context (for active processing)
    email_content: Optional[str] = None
    extracted_info: Optional[Dict[str, Any]] = None
    current_draft: Optional[str] = None
    draft_history: List[str] = field(default_factory=list)
    
    # Session history (preserves all emails and their data)
    email_sessions: List[EmailSession] = field(default_factory=list)
    
    # Currently viewed session (for operations like save on viewed sessions)
    currently_viewed_session: Optional[str] = None
    
    # General conversation context
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
    
    def archive_current_email_session(self):
        """Archive the current email session to preserve it in history"""
        if self.email_content:
            # Calculate drafts for this session only by excluding drafts from previous sessions
            previous_draft_count = sum(len(session.drafts) for session in self.email_sessions)
            current_session_drafts = self.draft_history[previous_draft_count:].copy()
            
            # Create a session record for the current email
            session = EmailSession(
                email_content=self.email_content,
                extracted_info=self.extracted_info.copy() if self.extracted_info else None,
                drafts=current_session_drafts,
                current_draft=self.current_draft,
                timestamp=datetime.now(),
                email_id=f"email_{len(self.email_sessions) + 1}"
            )
            self.email_sessions.append(session)
    
    def reset_email_context(self):
        """Reset email-specific context for processing a new email, preserving session history"""
        # Archive current session before resetting
        self.archive_current_email_session()
        
        # Reset current email context
        self.email_content = None
        self.extracted_info = None
        self.current_draft = None
        self.draft_history = []
        self.current_state = ConversationState.WAITING_FOR_EMAIL
    
    def get_all_session_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries of all email sessions in this conversation"""
        summaries = []
        for i, session in enumerate(self.email_sessions):
            summary = {
                'session_id': session.email_id or f"email_{i+1}",
                'timestamp': session.timestamp.isoformat(),
                'has_extracted_info': session.extracted_info is not None,
                'draft_count': len(session.drafts),
                'has_current_draft': session.current_draft is not None,
            }
            
            # Add email subject if available in extracted info
            if session.extracted_info and 'subject' in session.extracted_info:
                summary['subject'] = session.extracted_info['subject']
            
            # Add sender info if available
            if session.extracted_info and 'sender_name' in session.extracted_info:
                summary['sender'] = session.extracted_info['sender_name']
                
            summaries.append(summary)
        
        # Include current session if active
        if self.email_content:
            # Calculate drafts for current session only by excluding drafts from previous sessions
            previous_draft_count = sum(len(session.drafts) for session in self.email_sessions)
            current_session_draft_count = len(self.draft_history) - previous_draft_count
            
            current_summary = {
                'session_id': 'current',
                'timestamp': datetime.now().isoformat(),
                'has_extracted_info': self.extracted_info is not None,
                'draft_count': current_session_draft_count,
                'has_current_draft': self.current_draft is not None,
                'is_current': True
            }
            
            if self.extracted_info and 'subject' in self.extracted_info:
                current_summary['subject'] = self.extracted_info['subject']
            if self.extracted_info and 'sender_name' in self.extracted_info:
                current_summary['sender'] = self.extracted_info['sender_name']
                
            summaries.append(current_summary)
        
        return summaries
    
    def get_session_by_id(self, session_id: str) -> Optional[EmailSession]:
        """Get a specific email session by ID"""
        if session_id == 'current':
            if self.email_content:
                return EmailSession(
                    email_content=self.email_content,
                    extracted_info=self.extracted_info,
                    drafts=self.draft_history.copy(),
                    current_draft=self.current_draft,
                    email_id='current'
                )
            return None
        
        for session in self.email_sessions:
            if session.email_id == session_id:
                return session
        return None


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
                'VIEW_SESSION_HISTORY': ConversationState.GREETING,  # Stay in same state
                'VIEW_SPECIFIC_SESSION': ConversationState.GREETING,  # Stay in same state
            },
            ConversationState.WAITING_FOR_EMAIL: {
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,
                'GENERAL_HELP': ConversationState.WAITING_FOR_EMAIL,
                'VIEW_SESSION_HISTORY': ConversationState.WAITING_FOR_EMAIL,  # Stay in same state
                'VIEW_SPECIFIC_SESSION': ConversationState.WAITING_FOR_EMAIL,  # Stay in same state
            },
            ConversationState.EMAIL_LOADED: {
                'EXTRACT_INFO': ConversationState.INFO_EXTRACTED,
                'DRAFT_REPLY': ConversationState.DRAFT_CREATED,
                'CONTINUE_WORKFLOW': ConversationState.INFO_EXTRACTED,
                'DECLINE_OFFER': ConversationState.EMAIL_LOADED,  # Stay in same state
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,  # New email
                'VIEW_SESSION_HISTORY': ConversationState.EMAIL_LOADED,  # Stay in same state
                'VIEW_SPECIFIC_SESSION': ConversationState.EMAIL_LOADED,  # Stay in same state
            },
            ConversationState.INFO_EXTRACTED: {
                'DRAFT_REPLY': ConversationState.DRAFT_CREATED,
                'CONTINUE_WORKFLOW': ConversationState.DRAFT_CREATED,
                'DECLINE_OFFER': ConversationState.INFO_EXTRACTED,  # Stay in same state
                'EXTRACT_INFO': ConversationState.INFO_EXTRACTED,  # Allow re-showing info
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,  # New email
                'CLARIFICATION_NEEDED': ConversationState.INFO_EXTRACTED,  # Handle clarification requests
                'VIEW_SESSION_HISTORY': ConversationState.INFO_EXTRACTED,  # Stay in same state
                'VIEW_SPECIFIC_SESSION': ConversationState.INFO_EXTRACTED,  # Stay in same state
            },
            ConversationState.DRAFT_CREATED: {
                'REFINE_DRAFT': ConversationState.DRAFT_REFINED,
                'SAVE_DRAFT': ConversationState.READY_TO_SAVE,
                'CONTINUE_WORKFLOW': ConversationState.READY_TO_SAVE,
                'DECLINE_OFFER': ConversationState.DRAFT_CREATED,  # Stay in same state
                'DRAFT_REPLY': ConversationState.DRAFT_CREATED,  # New draft
                'EXTRACT_INFO': ConversationState.INFO_EXTRACTED,  # Allow transitioning to INFO_EXTRACTED for new emails
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,  # New email
                'VIEW_SESSION_HISTORY': ConversationState.DRAFT_CREATED,  # Stay in same state
                'VIEW_SPECIFIC_SESSION': ConversationState.DRAFT_CREATED,  # Stay in same state
            },
            ConversationState.DRAFT_REFINED: {
                'REFINE_DRAFT': ConversationState.DRAFT_REFINED,  # Multiple refinements
                'SAVE_DRAFT': ConversationState.READY_TO_SAVE,
                'CONTINUE_WORKFLOW': ConversationState.READY_TO_SAVE,
                'DECLINE_OFFER': ConversationState.DRAFT_REFINED,  # Stay in same state
                'DRAFT_REPLY': ConversationState.DRAFT_CREATED,  # Start over
                'EXTRACT_INFO': ConversationState.INFO_EXTRACTED,  # Allow transitioning to INFO_EXTRACTED for new emails
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,  # New email
                'VIEW_SESSION_HISTORY': ConversationState.DRAFT_REFINED,  # Stay in same state
                'VIEW_SPECIFIC_SESSION': ConversationState.DRAFT_REFINED,  # Stay in same state
            },
            ConversationState.READY_TO_SAVE: {
                'SAVE_DRAFT': ConversationState.CONVERSATION_COMPLETE,
                'REFINE_DRAFT': ConversationState.DRAFT_REFINED,  # More changes
                'DRAFT_REPLY': ConversationState.DRAFT_CREATED,  # New draft
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,  # New email
                'EXTRACT_INFO': ConversationState.INFO_EXTRACTED,  # Allow info extraction for new emails
                'VIEW_SESSION_HISTORY': ConversationState.READY_TO_SAVE,  # Stay in same state
                'VIEW_SPECIFIC_SESSION': ConversationState.READY_TO_SAVE,  # Stay in same state
            },
            ConversationState.CONVERSATION_COMPLETE: {
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,  # New email
                'GENERAL_HELP': ConversationState.GREETING,
                'VIEW_SESSION_HISTORY': ConversationState.CONVERSATION_COMPLETE,  # Stay in same state
                'VIEW_SPECIFIC_SESSION': ConversationState.CONVERSATION_COMPLETE,  # Stay in same state
            },
            ConversationState.ERROR_RECOVERY: {
                'LOAD_EMAIL': ConversationState.EMAIL_LOADED,
                'DRAFT_REPLY': ConversationState.DRAFT_CREATED,
                'SAVE_DRAFT': ConversationState.CONVERSATION_COMPLETE,  # Allow saving from error recovery
                'EXTRACT_INFO': ConversationState.INFO_EXTRACTED,  # Allow showing info from error recovery
                'REFINE_DRAFT': ConversationState.DRAFT_REFINED,  # Allow refining from error recovery
                'GENERAL_HELP': ConversationState.GREETING,
                'CLARIFICATION_NEEDED': ConversationState.ERROR_RECOVERY,
                'VIEW_SESSION_HISTORY': ConversationState.ERROR_RECOVERY,  # Stay in same state
                'VIEW_SPECIFIC_SESSION': ConversationState.ERROR_RECOVERY,  # Stay in same state
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