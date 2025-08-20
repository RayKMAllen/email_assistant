"""
Main conversational email agent that orchestrates intent classification,
state management, email processing, and response generation.
"""

from typing import Dict, Any, Tuple
import traceback

from assistant.llm_session import EmailLLMProcessor
from assistant.conversation_state import ConversationStateManager, ConversationState
from assistant.intent_classifier import HybridIntentClassifier, IntentResult
from assistant.response_generator import ConversationalResponseGenerator


class ConversationalEmailAgent:
    """
    Main conversational agent that processes user input and manages
    the complete email assistance workflow through natural conversation
    """
    
    def __init__(self):
        # Initialize core components
        self.email_processor = EmailLLMProcessor()
        self.state_manager = ConversationStateManager()
        self.intent_classifier = HybridIntentClassifier(email_processor=self.email_processor)
        self.response_generator = ConversationalResponseGenerator(self.state_manager)
        
        # Track conversation metrics
        self.conversation_count = 0
        self.successful_operations = 0
        self.failed_operations = 0
    
    def process_user_input(self, user_input: str) -> str:
        """
        Main entry point for processing user input and generating responses
        
        Args:
            user_input: The user's natural language input
            
        Returns:
            Conversational response with proactive guidance
        """
        try:
            self.conversation_count += 1
            
            # Add user input to conversation history
            self.state_manager.context.add_to_history("user", user_input)
            
            # Classify user intent
            intent_result = self.intent_classifier.classify(
                user_input, 
                self.state_manager.context
            )
            
            # Handle clarification needed case
            if intent_result.intent == 'CLARIFICATION_NEEDED':
                response = self.response_generator.generate_clarification_response(
                    user_input, 
                    self.state_manager.get_context_summary()
                )
                self.state_manager.context.add_to_history("assistant", response)
                return response
            
            # Execute the appropriate action based on intent
            operation_result, success = self._execute_intent(intent_result, user_input)
            
            # Update conversation state
            new_state = self.state_manager.transition_state(
                intent_result.intent, 
                success
            )
            
            # Track success/failure
            if success:
                self.successful_operations += 1
            else:
                self.failed_operations += 1
            
            # Generate conversational response with proactive guidance
            response = self.response_generator.generate_response(
                intent_result.intent,
                operation_result,
                success
            )
            
            # Add response to conversation history
            self.state_manager.context.add_to_history("assistant", response)
            
            return response
            
        except Exception as e:
            # Handle unexpected errors gracefully
            error_response = self._handle_unexpected_error(e, user_input)
            self.state_manager.context.add_to_history("assistant", error_response)
            self.failed_operations += 1
            return error_response
    
    def _execute_intent(self, intent_result: IntentResult, user_input: str) -> Tuple[Any, bool]:
        """
        Execute the appropriate action based on the classified intent
        
        Args:
            intent_result: Result from intent classification
            user_input: Original user input
            
        Returns:
            Tuple of (operation_result, success_boolean)
        """
        intent = intent_result.intent
        parameters = intent_result.parameters
        
        try:
            if intent == 'LOAD_EMAIL':
                return self._handle_load_email(parameters, user_input)
            
            elif intent == 'EXTRACT_INFO':
                return self._handle_extract_info()
            
            elif intent == 'DRAFT_REPLY':
                return self._handle_draft_reply(parameters)
            
            elif intent == 'REFINE_DRAFT':
                return self._handle_refine_draft(parameters, user_input)
            
            elif intent == 'SAVE_DRAFT':
                return self._handle_save_draft(parameters)
            
            elif intent == 'GENERAL_HELP':
                return self._handle_general_help()
            
            elif intent == 'CONTINUE_WORKFLOW':
                return self._handle_continue_workflow()
            
            else:
                return f"I'm not sure how to handle that request: {intent}", False
                
        except Exception as e:
            return f"Error executing {intent}: {str(e)}", False
    
    def _handle_load_email(self, parameters: Dict[str, Any], user_input: str) -> Tuple[Dict[str, Any], bool]:
        """Handle loading and processing an email"""
        try:
            # Extract email content from parameters or user input
            email_content = parameters.get('email_content')
            if not email_content:
                # If no email content in parameters, use the full user input
                email_content = user_input
            
            # Load the email using existing bedrock session
            self.email_processor.load_text(email_content)
            self.state_manager.update_context(email_content=email_content)
            
            # Automatically extract key information
            self.email_processor.extract_key_info()
            extracted_info = self.email_processor.key_info
            self.state_manager.update_context(extracted_info=extracted_info)
            
            return {
                'email_content': email_content,
                'extracted_info': extracted_info
            }, True
            
        except Exception as e:
            return {'error': str(e)}, False
    
    def _handle_extract_info(self) -> Tuple[Dict[str, Any], bool]:
        """Handle extracting key information from loaded email"""
        try:
            if not self.email_processor.text:
                return {'error': 'No email loaded to extract information from'}, False
            
            # Extract key info if not already done
            if not self.email_processor.key_info:
                self.email_processor.extract_key_info()
            
            extracted_info = self.email_processor.key_info
            self.state_manager.update_context(extracted_info=extracted_info)
            
            return extracted_info, True
            
        except Exception as e:
            return {'error': str(e)}, False
    
    def _handle_draft_reply(self, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Handle drafting a reply to the email"""
        try:
            if not self.email_processor.text:
                return {'error': 'No email loaded to draft a reply for'}, False
            
            # Get tone from parameters
            tone = parameters.get('tone')
            
            # Draft the reply
            draft = self.email_processor.draft_reply(tone=tone)
            self.state_manager.update_context(current_draft=draft)
            
            # Add to draft history
            self.state_manager.context.draft_history.append(draft)
            
            return {
                'draft': draft,
                'tone': tone
            }, True
            
        except Exception as e:
            return {'error': str(e)}, False
    
    def _handle_refine_draft(self, parameters: Dict[str, Any], user_input: str) -> Tuple[str, bool]:
        """Handle refining an existing draft"""
        try:
            if not self.email_processor.last_draft:
                return "No draft available to refine", False
            
            # Get refinement instructions from parameters or user input
            instructions = parameters.get('refinement_instructions', user_input)
            
            # Refine the draft
            refined_draft = self.email_processor.refine(instructions)
            self.state_manager.update_context(current_draft=refined_draft)
            
            # Add to draft history
            self.state_manager.context.draft_history.append(refined_draft)
            
            return refined_draft, True
            
        except Exception as e:
            return f"Error refining draft: {str(e)}", False
    
    def _handle_save_draft(self, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Handle saving the current draft"""
        try:
            if not self.email_processor.last_draft:
                return {'error': 'No draft available to save'}, False
            
            # Determine save location and method
            filepath = parameters.get('filepath')
            cloud = parameters.get('cloud', False)
            
            # Save the draft
            self.email_processor.save_draft(filepath=filepath, cloud=cloud)
            
            # Determine actual filepath for response
            if not filepath:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"drafts/draft_{timestamp}.txt"
            
            return {'filepath': filepath, 'cloud': cloud}, True
            
        except Exception as e:
            return {'error': str(e)}, False
    
    def _handle_general_help(self) -> Tuple[str, bool]:
        """Handle general help requests"""
        return "help_requested", True
    
    def _handle_continue_workflow(self) -> Tuple[str, bool]:
        """Handle workflow continuation based on current state"""
        current_state = self.state_manager.context.current_state
        
        try:
            if current_state == ConversationState.EMAIL_LOADED:
                # Auto-extract info and prepare for drafting
                if not self.email_processor.key_info:
                    self.email_processor.extract_key_info()
                    self.state_manager.update_context(extracted_info=self.email_processor.key_info)
                return self.email_processor.key_info, True
            
            elif current_state == ConversationState.INFO_EXTRACTED:
                # Auto-draft reply
                draft = self.email_processor.draft_reply()
                self.state_manager.update_context(current_draft=draft)
                self.state_manager.context.draft_history.append(draft)
                return {'draft': draft}, True
            
            elif current_state == ConversationState.DRAFT_CREATED:
                # Prepare for saving
                return "ready_to_save", True
            
            else:
                return "continue_acknowledged", True
                
        except Exception as e:
            return f"Error continuing workflow: {str(e)}", False
    
    def _handle_unexpected_error(self, error: Exception, user_input: str) -> str:
        """Handle unexpected errors gracefully"""
        error_msg = str(error)
        
        # Log the full traceback for debugging
        print(f"Unexpected error processing '{user_input}': {error_msg}")
        print(traceback.format_exc())
        
        # Set state to error recovery
        self.state_manager.context.current_state = ConversationState.ERROR_RECOVERY
        
        # Generate user-friendly error response
        friendly_responses = [
            "I encountered an unexpected issue. Let me know what you'd like to try next, and I'll do my best to help!",
            "Something went wrong there. Could you try rephrasing your request or let me know what you'd like to do?",
            "I ran into a problem processing that. What would you like me to help you with?",
        ]
        
        import random
        return random.choice(friendly_responses)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation state and metrics"""
        return {
            'conversation_state': self.state_manager.context.current_state.value,
            'conversation_count': self.conversation_count,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'has_email_loaded': self.state_manager.context.email_content is not None,
            'has_draft': self.state_manager.context.current_draft is not None,
            'draft_history_count': len(self.state_manager.context.draft_history),
            'conversation_history_length': len(self.state_manager.context.conversation_history),
        }
    
    def reset_conversation(self):
        """Reset the conversation state for a new session"""
        self.state_manager.context.reset_email_context()
        self.state_manager.context.current_state = ConversationState.GREETING
        self.conversation_count = 0
        self.successful_operations = 0
        self.failed_operations = 0
    
    def get_greeting_message(self) -> str:
        """Get the initial greeting message for new conversations"""
        greeting_messages = [
            "Hello! I'm your email assistant. I can help you process emails, extract key information, and draft professional replies. What can I help you with today?",
            "Hi there! I'm here to help you with your emails. I can process email content, extract important details, and help you draft replies. How can I assist you?",
            "Welcome! I'm your conversational email assistant. I can help you analyze emails and create professional responses. What would you like to work on?",
        ]
        
        import random
        return random.choice(greeting_messages)