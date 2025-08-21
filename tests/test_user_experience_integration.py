"""
User experience integration tests.
Tests realistic user scenarios, conversation flows, and usability aspects.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
from datetime import datetime

from src.assistant.conversational_agent import ConversationalEmailAgent
from assistant.conversation_state import ConversationState
from src.cli.cli import cli
from click.testing import CliRunner


class TestRealisticUserScenarios:
    """Test realistic user scenarios and workflows"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_new_user_onboarding_experience(self, mock_processor_class):
        """Test the experience of a new user learning the system"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # New user starts with greeting
        greeting = agent.get_greeting_message()
        assert any(keyword in greeting.lower() for keyword in ['hello', 'hi', 'welcome', 'assistant'])
        assert any(keyword in greeting.lower() for keyword in ['email', 'help', 'process'])
        
        # User asks what the system can do
        response1 = agent.process_user_input("What can you do?")
        assert any(keyword in response1.lower() for keyword in ['process', 'draft', 'extract', 'save'])
        assert any(keyword in response1.lower() for keyword in ['email', 'reply', 'information'])
        
        # User asks for help
        response2 = agent.process_user_input("How do I get started?")
        assert any(keyword in response2.lower() for keyword in ['email', 'paste', 'share', 'provide'])
        
        # User provides their first email
        email = """From: boss@company.com
To: user@company.com
Subject: Project Update Required
Dear Team Member,
Please provide an update on the current project status by end of week.
Best regards, Manager"""
        
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {
            'summary': 'Manager requesting project status update by end of week',
            'sender_name': 'Manager',
            'subject': 'Project Update Required',
            'action_required': 'Provide project status update'
        }
        mock_processor.text = email
        
        response3 = agent.process_user_input(f"Here's an email I need help with: {email}")
        
        # Should provide helpful guidance for next steps
        assert "processed" in response3.lower() or "loaded" in response3.lower()
        assert any(keyword in response3.lower() for keyword in ['draft', 'reply', 'extract', 'information'])
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_experienced_user_efficient_workflow(self, mock_processor_class):
        """Test efficient workflow for experienced users"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Experienced user provides email and immediately asks for draft
        email = "From: client@company.com\nSubject: Meeting Request\nCan we meet next week?"
        
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Meeting request for next week'}
        mock_processor.text = email
        mock_processor.draft_reply = Mock(return_value="I'd be happy to meet next week. What day works best?")
        mock_processor.last_draft = "I'd be happy to meet next week. What day works best?"
        
        response = agent.process_user_input(f"Process this email and draft a professional reply: {email}")
        
        # Should handle compound request efficiently
        assert "draft" in response.lower()
        assert "What day works best?" in response
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_user_changing_mind_workflow(self, mock_processor_class):
        """Test user changing their mind during workflow"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test email'}
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # User initially declines draft offer
        agent.state_manager.context.current_state = ConversationState.INFO_EXTRACTED
        agent.state_manager.context.email_content = "email content"
        
        response1 = agent.process_user_input("no thanks")
        assert any(keyword in response1.lower() for keyword in ['no problem', 'fine', 'what would you like'])
        
        # User changes mind and wants draft
        mock_processor.draft_reply = Mock(return_value="Changed mind draft")
        mock_processor.last_draft = "Changed mind draft"
        
        response2 = agent.process_user_input("actually, can you draft a reply?")
        assert "draft" in response2.lower()
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_user_iterative_refinement_experience(self, mock_processor_class):
        """Test user iteratively refining drafts"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "Initial draft"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        agent.state_manager.context.current_state = ConversationState.DRAFT_CREATED
        agent.state_manager.context.current_draft = "Initial draft"
        
        # Multiple refinement iterations
        refinements = [
            ("make it more formal", "Dear Sir/Madam, I am writing to..."),
            ("add a meeting request", "Dear Sir/Madam, I am writing to... Could we schedule a meeting?"),
            ("make it shorter", "Dear Sir/Madam, Could we schedule a meeting? Best regards."),
            ("add my contact info", "Dear Sir/Madam, Could we schedule a meeting? Contact me at user@email.com. Best regards.")
        ]
        
        for request, expected_draft in refinements:
            mock_processor.refine = Mock(return_value=expected_draft)
            mock_processor.last_draft = expected_draft
            
            response = agent.process_user_input(request)
            
            assert "refined" in response.lower() or "updated" in response.lower()
            assert expected_draft in response
            assert agent.state_manager.context.current_state == ConversationState.DRAFT_REFINED
        
        # Verify draft history is maintained
        assert len(agent.state_manager.context.draft_history) >= len(refinements)


class TestConversationalFlowExperience:
    """Test natural conversational flow and user experience"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_natural_language_understanding(self, mock_processor_class):
        """Test understanding of natural language variations"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test various ways users might express the same intent
        natural_variations = [
            # Draft requests
            ("write a response", "DRAFT_REPLY"),
            ("help me reply", "DRAFT_REPLY"),
            ("compose an answer", "DRAFT_REPLY"),
            ("I need to respond", "DRAFT_REPLY"),
            
            # Refinement requests
            ("make it sound better", "REFINE_DRAFT"),
            ("improve the tone", "REFINE_DRAFT"),
            ("polish it up", "REFINE_DRAFT"),
            ("can you fix this?", "REFINE_DRAFT"),
            
            # Save requests
            ("keep this", "SAVE_DRAFT"),
            ("store the draft", "SAVE_DRAFT"),
            ("I want to save it", "SAVE_DRAFT"),
            ("export this", "SAVE_DRAFT")
        ]
        
        for user_input, expected_intent in natural_variations:
            # Setup appropriate mocks based on intent
            if expected_intent == "DRAFT_REPLY":
                mock_processor.draft_reply = Mock(return_value="Draft response")
                mock_processor.last_draft = "Draft response"
            elif expected_intent == "REFINE_DRAFT":
                mock_processor.refine = Mock(return_value="Refined draft")
                mock_processor.last_draft = "Refined draft"
            elif expected_intent == "SAVE_DRAFT":
                mock_processor.save_draft = Mock()
            
            response = agent.process_user_input(user_input)
            
            # Should understand and respond appropriately
            assert isinstance(response, str)
            assert len(response) > 0
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_contextual_conversation_memory(self, mock_processor_class):
        """Test that system remembers context throughout conversation"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Load email with specific details
        email = """From: sarah@client.com
Subject: Budget Approval Needed
Hi, I need approval for the Q2 marketing budget of $50,000.
The deadline is next Friday. Thanks, Sarah"""
        
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {
            'sender_name': 'Sarah',
            'subject': 'Budget Approval Needed',
            'summary': 'Sarah requesting Q2 marketing budget approval of $50,000 by next Friday'
        }
        mock_processor.text = email
        
        agent.process_user_input(f"Process: {email}")
        
        # Later in conversation, user refers to context
        response1 = agent.process_user_input("What was Sarah asking for again?")
        # Should remember Sarah and the budget request
        assert "sarah" in response1.lower() or "budget" in response1.lower()
        
        # User asks about deadline
        response2 = agent.process_user_input("When does she need this by?")
        # Should remember the Friday deadline
        assert "friday" in response2.lower() or "deadline" in response2.lower()
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_proactive_guidance_experience(self, mock_processor_class):
        """Test proactive guidance and suggestions"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'meeting request'}
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test guidance in different states
        states_and_guidance = [
            (ConversationState.EMAIL_LOADED, ['extract', 'draft', 'information']),
            (ConversationState.INFO_EXTRACTED, ['draft', 'reply', 'response']),
            (ConversationState.DRAFT_CREATED, ['refine', 'save', 'formal', 'changes']),
            (ConversationState.DRAFT_REFINED, ['save', 'changes', 'satisfied'])
        ]
        
        for state, expected_keywords in states_and_guidance:
            agent.state_manager.context.current_state = state
            
            # Ask for general help to trigger guidance
            response = agent.process_user_input("what should I do next?")
            
            # Should provide relevant guidance
            assert any(keyword in response.lower() for keyword in expected_keywords)


class TestErrorRecoveryUserExperience:
    """Test user experience during error recovery"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_graceful_error_communication(self, mock_processor_class):
        """Test that errors are communicated gracefully to users"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Simulate various error scenarios
        error_scenarios = [
            ("Network timeout", "draft a reply"),
            ("Service unavailable", "extract information"),
            ("File not found", "save the draft"),
            ("Permission denied", "save to file")
        ]
        
        for error_msg, user_request in error_scenarios:
            # Setup error
            if "draft" in user_request:
                mock_processor.draft_reply = Mock(side_effect=Exception(error_msg))
            elif "extract" in user_request:
                mock_processor.extract_key_info = Mock(side_effect=Exception(error_msg))
            elif "save" in user_request:
                mock_processor.save_draft = Mock(side_effect=Exception(error_msg))
            
            response = agent.process_user_input(user_request)
            
            # Should communicate error in user-friendly way
            assert not any(technical_term in response.lower() for technical_term in [
                'traceback', 'exception', 'stack', 'null', 'undefined'
            ])
            assert any(friendly_term in response.lower() for friendly_term in [
                'problem', 'issue', 'trouble', 'try', 'help'
            ])
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_recovery_guidance_experience(self, mock_processor_class):
        """Test that users get helpful guidance during recovery"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Cause error
        mock_processor.draft_reply = Mock(side_effect=Exception("Service error"))
        response1 = agent.process_user_input("draft a reply")
        
        # Should be in error recovery state
        assert agent.state_manager.context.current_state == ConversationState.ERROR_RECOVERY
        
        # User asks what to do
        response2 = agent.process_user_input("what should I do now?")
        
        # Should provide helpful recovery guidance
        assert any(keyword in response2.lower() for keyword in [
            'try', 'again', 'help', 'different', 'alternative'
        ])


class TestAccessibilityAndUsability:
    """Test accessibility and usability aspects"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_clear_response_formatting(self, mock_processor_class):
        """Test that responses are clearly formatted and readable"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {
            'summary': 'Meeting request from John',
            'sender_name': 'John Doe',
            'subject': 'Weekly Meeting',
            'key_points': ['Schedule meeting', 'Discuss project']
        }
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test information extraction formatting
        response = agent.process_user_input("show me the key information")
        
        # Should have clear formatting
        assert "**" in response or "Summary:" in response or "From:" in response
        assert "John Doe" in response
        assert "Weekly Meeting" in response
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_consistent_terminology(self, mock_processor_class):
        """Test consistent terminology throughout the experience"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test that terminology is consistent
        responses = []
        
        # Draft creation
        mock_processor.draft_reply = Mock(return_value="Draft content")
        mock_processor.last_draft = "Draft content"
        responses.append(agent.process_user_input("create a draft"))
        
        # Draft refinement
        mock_processor.refine = Mock(return_value="Refined content")
        mock_processor.last_draft = "Refined content"
        responses.append(agent.process_user_input("refine the draft"))
        
        # Draft saving
        mock_processor.save_draft = Mock()
        responses.append(agent.process_user_input("save the draft"))
        
        # Check for consistent terminology
        draft_terms = ['draft', 'reply', 'response']
        for response in responses:
            # Should use consistent terms
            assert any(term in response.lower() for term in draft_terms)


class TestCLIUserExperience:
    """Test CLI user experience aspects"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('src.cli.cli.get_agent')
    def test_cli_help_accessibility(self, mock_get_agent, runner):
        """Test CLI help system accessibility"""
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        
        # Test help commands
        result = runner.invoke(cli, ['help-commands'])
        assert result.exit_code == 0
        
        # Should have clear sections
        assert "Natural Language Commands" in result.output
        assert "CLI Commands" in result.output
        assert "Tips" in result.output
        
        # Should have examples
        assert "example" in result.output.lower() or "'" in result.output
    
    @patch('src.cli.cli.get_agent')
    def test_cli_status_clarity(self, mock_get_agent, runner):
        """Test CLI status display clarity"""
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        mock_agent.get_conversation_summary.return_value = {
            'conversation_state': 'draft_created',
            'conversation_count': 5,
            'successful_operations': 4,
            'failed_operations': 1,
            'has_email_loaded': True,
            'has_draft': True,
            'draft_history_count': 2
        }
        
        result = runner.invoke(cli, ['status'])
        assert result.exit_code == 0
        
        # Should have clear status indicators
        assert "✅" in result.output or "❌" in result.output
        assert "Current State" in result.output
        assert "Messages Exchanged" in result.output
    
    @patch('src.cli.cli.get_agent')
    def test_cli_error_user_friendliness(self, mock_get_agent, runner):
        """Test CLI error handling user-friendliness"""
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        mock_agent.process_user_input.side_effect = Exception("Internal error")
        
        result = runner.invoke(cli, ['ask', 'test message'])
        assert result.exit_code == 0  # Should not crash
        
        # Should show user-friendly error
        assert "⚠️ Error:" in result.output
        assert "Internal error" in result.output


class TestMultiModalUserExperience:
    """Test multi-modal user experience (CLI + conversational)"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @patch('src.cli.cli.get_agent')
    def test_cli_conversational_mode_experience(self, mock_get_agent, runner):
        """Test conversational mode user experience"""
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        mock_agent.get_greeting_message.return_value = "Hello! I'm your email assistant."
        
        # Test conversational mode startup
        with patch('builtins.input', side_effect=['help', 'exit']):
            result = runner.invoke(cli, ['chat'])
        
        assert result.exit_code == 0
        assert "Hello! I'm your email assistant." in result.output
        assert "Tips:" in result.output
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_session_continuity_experience(self, mock_processor_class):
        """Test session continuity across different interaction modes"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test email'}
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Start session with email processing
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Important email'}
        mock_processor.text = "email content"
        
        agent.process_user_input("Process this email: content")
        
        # Check session state
        summary1 = agent.get_conversation_summary()
        assert summary1['has_email_loaded'] == True
        assert summary1['conversation_count'] == 1
        
        # Continue session
        mock_processor.draft_reply = Mock(return_value="Draft reply")
        mock_processor.last_draft = "Draft reply"
        
        agent.process_user_input("Draft a reply")
        
        # Verify continuity
        summary2 = agent.get_conversation_summary()
        assert summary2['has_email_loaded'] == True
        assert summary2['has_draft'] == True
        assert summary2['conversation_count'] == 2


class TestUserFeedbackAndAdaptation:
    """Test system adaptation to user feedback and preferences"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_learning_from_user_corrections(self, mock_processor_class):
        """Test system learning from user corrections and feedback"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "Initial draft"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        agent.state_manager.context.current_state = ConversationState.DRAFT_CREATED
        agent.state_manager.context.current_draft = "Initial draft"
        
        # User provides specific feedback
        feedback_scenarios = [
            "that's too formal, make it more casual",
            "add more details about the timeline",
            "remove the technical jargon",
            "make it sound more enthusiastic"
        ]
        
        for feedback in feedback_scenarios:
            mock_processor.refine = Mock(return_value=f"Refined based on: {feedback}")
            mock_processor.last_draft = f"Refined based on: {feedback}"
            
            response = agent.process_user_input(feedback)
            
            # Should acknowledge and incorporate feedback
            assert "refined" in response.lower() or "updated" in response.lower()
            assert feedback.split()[0] in response or "based on" in response
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_user_preference_recognition(self, mock_processor_class):
        """Test recognition of user preferences and patterns"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # User consistently requests formal tone
        formal_requests = [
            "draft a formal reply",
            "make it more professional",
            "use formal language"
        ]
        
        for request in formal_requests:
            mock_processor.draft_reply = Mock(return_value="Formal draft")
            mock_processor.refine = Mock(return_value="Formal refined draft")
            mock_processor.last_draft = "Formal draft"
            
            response = agent.process_user_input(request)
            
            # Should handle formal requests appropriately
            assert "formal" in response.lower() or "professional" in response.lower()