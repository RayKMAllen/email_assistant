"""
Natural language response generation system for the conversational email agent.
Generates contextual responses with proactive guidance based on conversation state.
"""

from typing import Dict, Any
import random

from assistant.conversation_state import ConversationState, ConversationStateManager


class ConversationalResponseGenerator:
    """
    Generates natural language responses with proactive guidance
    based on conversation state and intent results
    """
    
    def __init__(self, state_manager: ConversationStateManager):
        self.state_manager = state_manager
        self._setup_response_templates()
        self._setup_error_templates()
        self._setup_guidance_templates()
    
    def _setup_response_templates(self):
        """Define response templates for different intents and states"""
        self.response_templates = {
            'LOAD_EMAIL': {
                'success': [
                    "I've processed your email{email_info}. {summary}",
                    "Got it! I've loaded your email{email_info}. {summary}",
                    "Email processed successfully{email_info}. {summary}",
                ],
                'email_info_templates': [
                    " from {sender}",
                    " about {subject}",
                    " from {sender} about {subject}",
                    ""
                ]
            },
            'EXTRACT_INFO': {
                'success': [
                    "Here's the key information I extracted:",
                    "I've analyzed the email and found these details:",
                    "Here are the key details from your email:",
                ]
            },
            'DRAFT_REPLY': {
                'success': [
                    "I've drafted a reply for you{tone_info}:",
                    "Here's a draft response{tone_info}:",
                    "I've created a reply{tone_info}:",
                ],
                'tone_info_templates': {
                    'formal': " in a formal tone",
                    'casual': " in a casual tone",
                    'professional': " in a professional tone",
                    'friendly': " in a friendly tone",
                    'concise': " that's concise and to the point",
                }
            },
            'REFINE_DRAFT': {
                'success': [
                    "I've refined the draft based on your feedback:",
                    "Here's the updated version:",
                    "I've refined those changes to your draft:",
                ]
            },
            'SAVE_DRAFT': {
                'success': [
                    "Perfect! I've saved your draft to {filepath}.",
                    "Draft saved successfully to {filepath}.",
                    "Your draft has been saved to {filepath}.",
                ]
            },
            'GENERAL_HELP': {
                'success': [
                    "I'm your email assistant! Here's what I can help you with:",
                    "I can help you with several email-related tasks:",
                    "Here are my capabilities:",
                ]
            },
            'DECLINE_OFFER': {
                'success': [
                    "No problem! What would you like me to help you with instead?",
                    "That's fine! Let me know what else I can do for you.",
                    "Understood! What would you prefer to do next?",
                ]
            },
            'VIEW_SESSION_HISTORY': {
                'success': [
                    "Here's your session history:",
                    "I've processed these emails in our conversation:",
                    "Here are all the emails we've worked on:",
                ]
            },
            'VIEW_SPECIFIC_SESSION': {
                'success': [
                    "Here are the details for {session_id}:",
                    "Here's what we did with {session_id}:",
                    "Details for {session_id}:",
                ]
            }
        }
    
    def _setup_error_templates(self):
        """Define error response templates"""
        self.error_templates = {
            'LOAD_EMAIL': [
                "I had trouble processing that email. Could you try pasting the email content again, or if it's a file, make sure the path is correct? I can help you with that.",
                "I ran into a problem loading that email. Please check if the file path is correct or try pasting the email content directly, and I'll help you get it working.",
            ],
            'DRAFT_REPLY': [
                "I'm having trouble drafting a reply right now. This might be because the email content isn't loaded yet. Would you like me to help you share the email first?",
                "I encountered an issue drafting a reply. I need to have an email loaded before I can draft a response. Could you help me by sharing the email content?",
            ],
            'EXTRACT_INFO': [
                "I ran into a problem extracting the key information from that email. The format might be unusual. Could you try sharing the email content again so I can help you?",
                "I had trouble analyzing that email. Could you try repasting the email content or check if it's formatted correctly? I'm here to help you get this working.",
            ],
            'SAVE_DRAFT': [
                "I'm having trouble saving the draft right now. Would you like me to try saving it to a different location? I can help you find another approach.",
                "There was an issue saving your draft. Let me try a different approach or you can copy the content manually. I'm here to help you resolve this problem.",
            ],
            'GENERAL': [
                "I encountered an issue with that request. Let me know how you'd like to proceed, and I'll do my best to help you!",
                "I ran into a problem there. Could you try rephrasing your request or let me know what you'd like to do? I'm here to help!",
            ]
        }
    
    def _setup_guidance_templates(self):
        """Define proactive guidance templates for each state"""
        self.guidance_templates = {
            ConversationState.GREETING: [
                "I can help you process emails, extract key information, and draft professional replies. You can paste an email directly, provide a file path, or ask me what I can do!",
                "What can I help you with today? I can process emails, extract information, and help you draft replies. Just share an email or ask me about my capabilities!",
            ],
            ConversationState.WAITING_FOR_EMAIL: [
                "I'm ready to help you with your email! Please share the email content, provide a file path, or paste the email text directly.",
                "Please share the email you'd like me to process. You can paste the content, provide a file path, or upload a document.",
                "I'm waiting for your email. You can share it by pasting the content or providing a file path to the email document.",
            ],
            ConversationState.EMAIL_LOADED: [
                "Would you like me to extract the key information and draft a reply for you?",
                "I can now extract the key details and help you draft a response. Shall I proceed?",
                "What would you like me to do with this email? I can extract key information, draft a reply, or both!",
            ],
            ConversationState.INFO_EXTRACTED: [
                "Shall I draft a reply for you? I can make it formal, casual, or match any specific tone you prefer.",
                "Would you like me to create a draft response? I can adjust the tone to be formal, friendly, or however you'd like.",
                "Ready to draft a reply? Just let me know what tone you'd prefer, or I can use a professional default.",
            ],
            ConversationState.DRAFT_CREATED: [
                "How does this look? I can refine it to be more formal, concise, friendly, or make any other adjustments you'd like. Or if you're happy with it, I can save it for you.",
                "What do you think of this draft? I can make it more professional, add specific details, change the tone, or save it as-is.",
                "Does this draft work for you? I can refine it further or save it to a file when you're ready.",
            ],
            ConversationState.DRAFT_REFINED: [
                "How's this version? I can make additional changes if needed, or save it when you're satisfied.",
                "Is this better? I can continue refining it or save the draft when you're happy with it.",
                "Does this revised version work better? Let me know if you want more changes or if you're ready to save it.",
            ],
            ConversationState.READY_TO_SAVE: [
                "Your draft is ready! I can save it to a local file or upload it to your S3 bucket. Would you like me to save it now?",
                "Perfect! Shall I save this draft for you? I can save it locally or to the cloud.",
                "This draft looks good to go! Would you like me to save it to a file?",
            ],
            ConversationState.CONVERSATION_COMPLETE: [
                "Great! Is there anything else I can help you with? I can process another email or assist with any other email-related tasks.",
                "All done! Do you have another email you'd like me to help with, or is there anything else I can do?",
                "Perfect! Feel free to share another email if you need help with more correspondence.",
            ],
            ConversationState.ERROR_RECOVERY: [
                "Let's try that again. What would you like me to help you with?",
                "No worries! What can I help you with next?",
                "Let me know what you'd like to do, and I'll give it another try.",
            ]
        }
    
    def generate_response(self, intent: str, operation_result: Any, success: bool = True) -> str:
        """
        Generate a complete conversational response with proactive guidance
        
        Args:
            intent: The classified intent
            operation_result: Result from the operation (email content, draft, etc.)
            success: Whether the operation was successful
            
        Returns:
            Complete conversational response string
        """
        if not success:
            return self._generate_error_response(intent, operation_result)
        
        # Generate main response based on intent
        main_response = self._generate_main_response(intent, operation_result)
        
        # For SAVE_DRAFT, check if the save actually completed successfully
        # If it did, show completion guidance instead of ready-to-save guidance
        if intent == 'SAVE_DRAFT':
            # Check if the save operation actually completed by looking at the result
            if isinstance(operation_result, dict) and 'filepath' in operation_result:
                # Save completed successfully - show completion guidance
                completion_guidance = self._generate_completion_guidance()
                if main_response and completion_guidance:
                    return f"{main_response}\n\n{completion_guidance}"
                elif main_response:
                    return main_response
                else:
                    return completion_guidance or "Draft saved successfully!"
            else:
                # Save didn't complete (maybe just preparing to save) - show normal guidance
                guidance = self._generate_proactive_guidance()
                if main_response and guidance:
                    return f"{main_response}\n\n{guidance}"
                elif main_response:
                    return main_response
                else:
                    return guidance or "I'm here to help! What would you like me to do?"
        
        # Add proactive guidance for other intents
        guidance = self._generate_proactive_guidance()
        
        # Combine responses
        if main_response and guidance:
            return f"{main_response}\n\n{guidance}"
        elif main_response:
            return main_response
        else:
            return guidance or "I'm here to help! What would you like me to do?"
    
    def _generate_main_response(self, intent: str, operation_result: Any) -> str:
        """Generate the main response based on intent and result"""
        # Special handling for CONTINUE_WORKFLOW that results in draft creation
        if intent == 'CONTINUE_WORKFLOW' and isinstance(operation_result, dict) and 'draft' in operation_result:
            # Treat this as a draft reply response
            draft_templates = self.response_templates['DRAFT_REPLY']['success']
            template = random.choice(draft_templates)
            return self._format_draft_reply_response(template, operation_result)
        
        if intent not in self.response_templates:
            return ""
        
        templates = self.response_templates[intent]['success']
        template = random.choice(templates)
        
        # Format template based on intent type
        if intent == 'LOAD_EMAIL':
            return self._format_load_email_response(template, operation_result)
        elif intent == 'EXTRACT_INFO':
            return self._format_extract_info_response(template, operation_result)
        elif intent == 'DRAFT_REPLY':
            return self._format_draft_reply_response(template, operation_result)
        elif intent == 'REFINE_DRAFT':
            return self._format_refine_response(template, operation_result)
        elif intent == 'SAVE_DRAFT':
            return self._format_save_response(template, operation_result)
        elif intent == 'GENERAL_HELP':
            return self._format_help_response(template)
        elif intent == 'DECLINE_OFFER':
            return self._format_decline_response(template, operation_result)
        elif intent == 'VIEW_SESSION_HISTORY':
            return self._format_session_history_response(template, operation_result)
        elif intent == 'VIEW_SPECIFIC_SESSION':
            return self._format_specific_session_response(template, operation_result)
        else:
            return template
    
    def _format_load_email_response(self, template: str, result: Dict[str, Any]) -> str:
        """Format response for email loading"""
        email_info = ""
        summary = ""
        
        if isinstance(result, dict):
            extracted_info = result.get('extracted_info', {})
            if extracted_info:
                sender = extracted_info.get('sender_name', '')
                subject = extracted_info.get('subject', '')
                
                if sender and subject:
                    email_info = f" from {sender} about {subject}"
                elif sender:
                    email_info = f" from {sender}"
                elif subject:
                    email_info = f" about {subject}"
                
                # If info was auto-extracted, include it in the response
                if result.get('auto_extracted') and 'summary' in extracted_info:
                    summary = f"I've also extracted the key information. Here's a quick summary: {extracted_info['summary']}"
                elif 'summary' in extracted_info:
                    summary = f"Here's a quick summary: {extracted_info['summary']}"
            
            # Check if this was a compound request that also created a draft
            if result.get('compound_request') and 'draft' in result:
                # Format as a draft response instead of just email loading
                base_response = template.format(email_info=email_info, summary=summary)
                draft_content = result['draft']
                tone_info = ""
                
                if result.get('tone'):
                    tone = result['tone']
                    tone_templates = {
                        'formal': " in a formal tone",
                        'casual': " in a casual tone",
                        'professional': " in a professional tone",
                        'friendly': " in a friendly tone",
                        'concise': " that's concise and to the point",
                    }
                    tone_info = tone_templates.get(tone, f" in a {tone} tone")
                
                return f"{base_response} I've also drafted a reply{tone_info}:\n\n{draft_content}"
        
        return template.format(email_info=email_info, summary=summary)
    
    def _format_extract_info_response(self, template: str, result: Dict[str, Any]) -> str:
        """Format response for information extraction"""
        # Handle case where info was already extracted
        if isinstance(result, dict) and result.get('already_extracted'):
            template = "Here's the key information I extracted earlier:"
            result = result.get('extracted_info', {})
        
        response = template
        
        if isinstance(result, dict):
            # Format the extracted information nicely
            info_lines = []
            for key, value in result.items():
                if key == 'summary':
                    info_lines.append(f"**Summary:** {value}")
                elif key == 'sender_name':
                    info_lines.append(f"**From:** {value}")
                elif key == 'receiver_name':
                    info_lines.append(f"**To:** {value}")
                elif key == 'subject':
                    info_lines.append(f"**Subject:** {value}")
                elif key == 'sender_contact_details':
                    if isinstance(value, dict):
                        contact_info = ", ".join([f"{k}: {v}" for k, v in value.items()])
                        info_lines.append(f"**Sender Contact:** {contact_info}")
                    elif isinstance(value, str):
                        info_lines.append(f"**Sender Contact:** {value}")
                elif key == 'receiver_contact_details':
                    if isinstance(value, dict):
                        contact_info = ", ".join([f"{k}: {v}" for k, v in value.items()])
                        info_lines.append(f"**Receiver Contact:** {contact_info}")
                    elif isinstance(value, str):
                        info_lines.append(f"**Receiver Contact:** {value}")
            
            if info_lines:
                response += "\n\n" + "\n".join(info_lines)
        
        return response
    
    def _format_draft_reply_response(self, template: str, result: Dict[str, Any]) -> str:
        """Format response for draft reply"""
        tone_info = ""
        
        if isinstance(result, dict) and 'tone' in result:
            tone = result['tone']
            tone_templates = self.response_templates['DRAFT_REPLY'].get('tone_info_templates', {})
            tone_info = tone_templates.get(tone, f" in a {tone} tone")
        
        formatted_template = template.format(tone_info=tone_info)
        
        # Add the actual draft content
        if isinstance(result, dict) and 'draft' in result:
            return f"{formatted_template}\n\n{result['draft']}"
        elif isinstance(result, str):
            return f"{formatted_template}\n\n{result}"
        
        return formatted_template
    
    def _format_refine_response(self, template: str, result: str) -> str:
        """Format response for draft refinement"""
        if isinstance(result, str):
            return f"{template}\n\n{result}"
        return template
    
    def _format_save_response(self, template: str, result: Dict[str, Any]) -> str:
        """Format response for saving draft"""
        filepath = "the default location"
        
        if isinstance(result, dict) and 'filepath' in result:
            filepath = result['filepath']
        elif isinstance(result, str):
            filepath = result
        
        return template.format(filepath=filepath)
    
    def _format_help_response(self, template: str) -> str:
        """Format help response with capabilities list"""
        capabilities = [
            "📧 **Process emails** - Load from text, file paths, or PDF files",
            "🔍 **Extract key information** - Get sender, receiver, subject, and summary",
            "✍️ **Draft replies** - Create professional responses with customizable tone",
            "🔧 **Refine drafts** - Make them more formal, casual, concise, or add specific content",
            "💾 **Save drafts** - Export to local files or cloud storage",
            "🔄 **Iterative refinement** - Keep improving until you're satisfied"
        ]
        
        return f"{template}\n\n" + "\n".join(capabilities)
    
    def _format_decline_response(self, template: str, result: str) -> str:
        """Format response for declined offers"""
        if result == "offer_declined_draft":
            return "No problem! You can ask me to draft a reply later, or I can help you with something else. What would you like to do?"
        elif result == "offer_declined_save":
            return "That's fine! You can save the draft later, make more changes, or I can help you with something else. What would you prefer?"
        elif result == "offer_declined_general":
            return template
        else:
            return template
    
    def _format_session_history_response(self, template: str, result: Dict[str, Any]) -> str:
        """Format response for session history"""
        if not isinstance(result, dict) or 'session_summaries' not in result:
            return template + "\n\nNo sessions found."
        
        sessions = result['session_summaries']
        if not sessions:
            return template + "\n\nNo emails have been processed in this conversation yet."
        
        response = template + "\n"
        
        for i, session in enumerate(sessions, 1):
            session_line = f"\n**{i}. "
            
            if session.get('is_current'):
                session_line += "Current Email"
            else:
                session_line += f"Email {i}"
            
            if 'subject' in session:
                session_line += f"** - {session['subject']}"
            else:
                session_line += "**"
            
            if 'sender' in session:
                session_line += f" (from {session['sender']})"
            
            session_line += f"\n  - Processed: {session['timestamp'][:19].replace('T', ' ')}"
            session_line += f"\n  - Drafts created: {session['draft_count']}"
            
            if session.get('has_extracted_info'):
                session_line += "\n  - Key information extracted ✓"
            
            if session.get('has_current_draft'):
                session_line += "\n  - Has current draft ✓"
            
            response += session_line
        
        response += f"\n\nTotal sessions: {result['total_sessions']}"
        response += "\n\nYou can view details of any session by saying 'show email [number]' or 'view session [number]'."
        
        return response
    
    def _format_specific_session_response(self, template: str, result: Dict[str, Any]) -> str:
        """Format response for specific session details"""
        if not isinstance(result, dict) or 'session' not in result:
            return template + "\n\nSession not found."
        
        session = result['session']
        session_id = session.get('session_id', 'Unknown')
        
        response = template.format(session_id=session_id) + "\n"
        
        # Add timestamp
        if 'timestamp' in session:
            response += f"\n**Processed:** {session['timestamp'][:19].replace('T', ' ')}"
        
        # Add extracted info if available
        if session.get('extracted_info'):
            info = session['extracted_info']
            response += "\n\n**Key Information:**"
            
            if 'sender_name' in info:
                response += f"\n- **From:** {info['sender_name']}"
            if 'subject' in info:
                response += f"\n- **Subject:** {info['subject']}"
            if 'summary' in info:
                response += f"\n- **Summary:** {info['summary']}"
        
        # Add draft information
        draft_count = session.get('draft_count', 0)
        if draft_count > 0:
            response += f"\n\n**Drafts Created:** {draft_count}"
            
            if session.get('current_draft'):
                response += "\n\n**Current Draft:**\n"
                response += session['current_draft']
        
        # Add email content (truncated)
        if session.get('email_content'):
            email_preview = session['email_content'][:200]
            if len(session['email_content']) > 200:
                email_preview += "..."
            response += f"\n\n**Email Content (preview):**\n{email_preview}"
        
        return response
    
    def _generate_proactive_guidance(self) -> str:
        """Generate proactive guidance based on current conversation state"""
        current_state = self.state_manager.context.current_state
        
        if current_state not in self.guidance_templates:
            return "What would you like me to help you with next?"
        
        templates = self.guidance_templates[current_state]
        return random.choice(templates)
    
    def _generate_completion_guidance(self) -> str:
        """Generate appropriate guidance after completing a save operation"""
        completion_templates = [
            "Great! Is there anything else I can help you with? I can process another email or assist with any other email-related tasks.",
            "All done! Do you have another email you'd like me to help with, or is there anything else I can do?",
            "Perfect! Feel free to share another email if you need help with more correspondence.",
        ]
        return random.choice(completion_templates)
    
    def _generate_error_response(self, intent: str, error_details: Any) -> str:
        """Generate helpful error responses that maintain conversational flow"""
        templates = self.error_templates.get(intent, self.error_templates['GENERAL'])
        base_response = random.choice(templates)
        
        # Add specific error context if available
        if isinstance(error_details, dict) and 'error' in error_details:
            return f"{base_response}\n\nError details: {error_details['error']}"
        
        return base_response
    
    def generate_clarification_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate response when user intent needs clarification"""
        # Check if this was a fallback from failed intent classification
        fallback_attempted = context.get('fallback_attempted', False)
        
        if fallback_attempted:
            clarification_templates = [
                "I tried to understand your request but I'm not quite sure what you'd like me to do. Let me help you with some options:",
                "I had trouble interpreting that request. Here are some things I can help you with:",
                "I'm not certain what you're asking for. Let me show you what I can do:",
            ]
        else:
            clarification_templates = [
                "I'd be happy to help! Could you clarify what you'd like me to do?",
                "I want to make sure I understand correctly. What would you like me to help you with?",
                "I'm not quite sure what you need. Could you be more specific?",
            ]
        
        base_response = random.choice(clarification_templates)
        
        # Add contextual suggestions based on current state
        current_state = self.state_manager.context.current_state
        
        suggestions = []
        if current_state == ConversationState.GREETING:
            suggestions = [
                "- Share an email you'd like me to process",
                "- Ask me what I can do",
                "- Provide a file path to an email document"
            ]
        elif current_state == ConversationState.EMAIL_LOADED or current_state == ConversationState.INFO_EXTRACTED:
            suggestions = [
                "- Ask me to extract key information or show summary",
                "- Request a draft reply",
                "- Ask for specific details about the email"
            ]
        elif current_state == ConversationState.DRAFT_CREATED or current_state == ConversationState.DRAFT_REFINED:
            suggestions = [
                "- Ask me to refine the draft (make it more formal, casual, etc.)",
                "- Request to save the draft",
                "- Ask for specific changes to the content"
            ]
        elif current_state == ConversationState.READY_TO_SAVE:
            suggestions = [
                "- Save the draft locally or to cloud",
                "- Make more refinements to the draft",
                "- Start working on a new email"
            ]
        
        if suggestions:
            suggestion_text = "For example, you could:\n" + "\n".join(suggestions)
            return f"{base_response}\n\n{suggestion_text}"
        
        return base_response