"""
Advanced integration tests for complex workflows and edge cases.
Tests advanced scenarios, error recovery, and cross-component interactions.
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


class TestAdvancedWorkflowIntegration:
    """Test advanced workflow scenarios and edge cases"""
    
    @pytest.fixture
    def complex_email(self):
        """Complex email with multiple recipients, attachments, and formatting"""
        return """From: project.manager@company.com
To: team-lead@company.com, developer1@company.com, developer2@company.com
Cc: stakeholder@client.com
Subject: URGENT: Project Deadline Extension Request - Action Required
Date: Mon, 15 Jan 2024 14:30:00 +0000
Priority: High

Dear Team,

I hope this email finds you well. I'm writing to discuss the current status of Project Alpha and request a deadline extension.

**Current Situation:**
- Original deadline: January 20, 2024
- Current progress: 75% complete
- Remaining tasks: Testing, documentation, deployment

**Challenges Faced:**
1. Unexpected technical issues with the authentication module
2. Key team member illness (John out for 1 week)
3. Client requested additional features mid-project

**Proposed Solution:**
I would like to request a 2-week extension to February 3, 2024. This will allow us to:
- Complete thorough testing
- Implement the additional client requirements
- Ensure proper documentation

**Next Steps:**
Please review this request and let me know your thoughts by EOD Wednesday. We can schedule a meeting to discuss if needed.

Best regards,
Sarah Johnson
Project Manager
sarah.johnson@company.com
Phone: (555) 123-4567
Mobile: (555) 987-6543"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_complex_email_workflow(self, mock_processor_class, complex_email):
        """Test processing complex email with multiple recipients and detailed content"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Setup complex extracted info
        complex_info = {
            'summary': 'Project deadline extension request due to technical issues and team challenges',
            'sender_name': 'Sarah Johnson',
            'sender_email': 'project.manager@company.com',
            'receiver_name': 'Team Lead',
            'receiver_email': 'team-lead@company.com',
            'subject': 'URGENT: Project Deadline Extension Request - Action Required',
            'priority': 'High',
            'deadline': 'January 20, 2024',
            'requested_extension': 'February 3, 2024',
            'key_points': [
                'Project Alpha 75% complete',
                'Technical issues with authentication module',
                'Team member illness affecting timeline',
                'Client requested additional features',
                'Requesting 2-week extension'
            ],
            'action_required': 'Review request and respond by EOD Wednesday',
            'sender_contact_details': {
                'email': 'sarah.johnson@company.com',
                'phone': '(555) 123-4567',
                'mobile': '(555) 987-6543',
                'title': 'Project Manager'
            }
        }
        
        # Load complex email
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = complex_info
        mock_processor.text = complex_email
        
        response1 = agent.process_user_input(f"Process this urgent email: {complex_email}")
        
        assert "processed" in response1.lower() or "loaded" in response1.lower()
        assert agent.state_manager.context.current_state == ConversationState.INFO_EXTRACTED
        
        # Draft professional response acknowledging complexity
        mock_processor.draft_reply = Mock(return_value="""Dear Sarah,

Thank you for your detailed email regarding the Project Alpha deadline extension request.

I understand the challenges you've outlined:
- Technical issues with the authentication module
- Team member absence due to illness
- Additional client requirements

The requested extension to February 3, 2024 seems reasonable given the circumstances. I'll review this with the stakeholders and provide a response by Wednesday as requested.

In the meantime, please keep me updated on the progress and let me know if you need any additional resources.

Best regards,
Team Lead""")
        mock_processor.last_draft = mock_processor.draft_reply.return_value
        
        response2 = agent.process_user_input("Draft a professional response acknowledging the complexity")
        
        assert "draft" in response2.lower()
        assert "authentication module" in response2 or "technical issues" in response2
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
        
        # Refine to add specific timeline commitments
        mock_processor.refine = Mock(return_value="""Dear Sarah,

Thank you for your comprehensive email regarding the Project Alpha deadline extension request.

After reviewing the challenges you've outlined - the authentication module issues, John's absence, and the additional client requirements - I believe the requested extension to February 3, 2024 is justified.

**My Response:**
- Extension approved in principle
- Will confirm with stakeholders by Wednesday EOD as requested
- Suggest we schedule a progress review meeting for Friday

**Additional Support:**
- I can assign a backup developer to assist with the authentication module
- Let's discuss the client requirements to ensure they're properly scoped

Please send me the current project status report and we'll move forward with the revised timeline.

Best regards,
Team Lead""")
        mock_processor.last_draft = mock_processor.refine.return_value
        
        response3 = agent.process_user_input("Add specific commitments and offer additional support")
        
        assert "refined" in response3.lower() or "updated" in response3.lower()
        assert "backup developer" in response3 or "additional support" in response3
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_REFINED
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_multi_email_session_workflow(self, mock_processor_class):
        """Test handling multiple emails in a single session with context switching"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Process first email - meeting request
        email1 = "From: alice@company.com\nSubject: Team Meeting\nDear Team, let's meet tomorrow at 2 PM."
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Team meeting request for tomorrow at 2 PM', 'sender_name': 'Alice'}
        mock_processor.text = email1
        
        response1 = agent.process_user_input(f"Process this email: {email1}")
        assert agent.state_manager.context.current_state == ConversationState.INFO_EXTRACTED
        
        # Draft reply for first email
        mock_processor.draft_reply = Mock(return_value="I'll be there. Thanks for organizing!")
        mock_processor.last_draft = "I'll be there. Thanks for organizing!"
        
        response2 = agent.process_user_input("Draft a quick acceptance reply")
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
        
        # Process second email - this should archive the first session
        email2 = "From: bob@company.com\nSubject: Budget Review\nWe need to review Q1 budget numbers."
        mock_processor.key_info = {'summary': 'Q1 budget review request', 'sender_name': 'Bob'}
        mock_processor.text = email2
        
        response3 = agent.process_user_input(f"Now process this different email: {email2}")
        assert agent.state_manager.context.current_state == ConversationState.INFO_EXTRACTED
        
        # Verify session history contains both emails
        history_response = agent.process_user_input("Show me the session history")
        assert "session" in history_response.lower() or "email" in history_response.lower()
        
        # Verify we can view specific sessions
        view_response = agent.process_user_input("Show me email 1")
        assert "alice" in view_response.lower() or "meeting" in view_response.lower()
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_error_recovery_and_retry_workflow(self, mock_processor_class):
        """Test comprehensive error recovery scenarios"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Successful email loading
        email = "From: test@example.com\nSubject: Test\nTest content"
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Test email'}
        mock_processor.text = email
        
        response1 = agent.process_user_input(f"Here's an email I need help with: {email}")
        assert agent.state_manager.context.current_state == ConversationState.INFO_EXTRACTED
        
        # First draft attempt fails
        mock_processor.draft_reply = Mock(side_effect=Exception("LLM service temporarily unavailable"))
        
        response2 = agent.process_user_input("Draft a reply")
        assert agent.state_manager.context.current_state == ConversationState.ERROR_RECOVERY
        assert "error" in response2.lower() or "problem" in response2.lower()
        
        # Second draft attempt also fails
        mock_processor.draft_reply = Mock(side_effect=Exception("Network timeout"))
        
        response3 = agent.process_user_input("Try drafting again")
        assert agent.state_manager.context.current_state == ConversationState.ERROR_RECOVERY
        
        # Third attempt succeeds
        mock_processor.draft_reply = Mock(return_value="Thank you for your email. I'll get back to you soon.")
        mock_processor.last_draft = "Thank you for your email. I'll get back to you soon."
        
        response4 = agent.process_user_input("Please try the draft one more time")
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
        assert "draft" in response4.lower()
        
        # Verify error recovery metrics
        assert agent.failed_operations >= 2
        assert agent.successful_operations >= 2
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_conversation_context_preservation(self, mock_processor_class):
        """Test that conversation context is preserved across complex interactions"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Load email with specific context
        email = """From: client@bigcorp.com
Subject: Contract Renewal Discussion
Dear Partner,
We need to discuss the renewal of our service contract which expires next month.
The current terms have been working well, but we'd like to explore some modifications.
Best regards, John Smith, Procurement Manager"""
        
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {
            'summary': 'Contract renewal discussion with potential modifications',
            'sender_name': 'John Smith',
            'sender_email': 'client@bigcorp.com',
            'subject': 'Contract Renewal Discussion',
            'key_points': ['Contract expires next month', 'Current terms working well', 'Want to explore modifications']
        }
        mock_processor.text = email
        
        agent.process_user_input(f"Process this important email: {email}")
        
        # Draft initial response
        mock_processor.draft_reply = Mock(return_value="Thank you for reaching out about the contract renewal.")
        mock_processor.last_draft = "Thank you for reaching out about the contract renewal."
        
        agent.process_user_input("Draft a professional reply")
        
        # Multiple refinements that should maintain context
        mock_processor.refine = Mock(return_value="Thank you for reaching out about the contract renewal. I'm pleased to hear the current terms have been working well.")
        mock_processor.last_draft = mock_processor.refine.return_value
        
        agent.process_user_input("Add acknowledgment of their satisfaction")
        
        mock_processor.refine = Mock(return_value="Thank you for reaching out about the contract renewal. I'm pleased to hear the current terms have been working well. I'd be happy to schedule a meeting to discuss potential modifications.")
        mock_processor.last_draft = mock_processor.refine.return_value
        
        agent.process_user_input("Offer to schedule a meeting")
        
        # Verify context preservation
        context = agent.state_manager.context
        assert context.email_content is not None
        assert context.extracted_info is not None
        assert context.current_draft is not None
        assert len(context.draft_history) >= 2
        assert len(context.conversation_history) >= 6  # Multiple exchanges
        
        # Verify specific context details are maintained
        assert 'John Smith' in str(context.extracted_info)
        assert 'contract renewal' in context.current_draft.lower()


class TestIntentClassificationIntegration:
    """Test intent classification in complex scenarios"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_ambiguous_intent_resolution(self, mock_processor_class):
        """Test handling of ambiguous user inputs that require clarification"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "Sample email content"
        mock_processor.key_info = {'summary': 'Test email'}
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Set up email loaded state
        agent.state_manager.context.current_state = ConversationState.EMAIL_LOADED
        agent.state_manager.context.email_content = "Sample email"
        
        # Test ambiguous inputs
        ambiguous_inputs = [
            "do something",
            "help",
            "what now",
            "fix it",
            "make changes"
        ]
        
        for ambiguous_input in ambiguous_inputs:
            response = agent.process_user_input(ambiguous_input)
            # Should either provide clarification or specific guidance
            assert any(keyword in response.lower() for keyword in [
                'clarify', 'specific', 'help', 'can', 'would you like', 'for example'
            ])
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_context_dependent_intent_classification(self, mock_processor_class):
        """Test that intent classification considers conversation context"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "Email content"
        mock_processor.key_info = {'summary': 'Test'}
        mock_processor.last_draft = "Draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test "yes" in different contexts
        
        # Context 1: After email loaded, "yes" should continue workflow
        agent.state_manager.context.current_state = ConversationState.EMAIL_LOADED
        agent.state_manager.context.email_content = "Email content"
        
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Test email'}
        
        response1 = agent.process_user_input("yes")
        # Should extract info or draft reply
        assert agent.state_manager.context.current_state in [
            ConversationState.INFO_EXTRACTED, ConversationState.DRAFT_CREATED
        ]
        
        # Context 2: After draft created, "yes" should save or continue
        agent.state_manager.context.current_state = ConversationState.DRAFT_CREATED
        agent.state_manager.context.current_draft = "Draft content"
        
        response2 = agent.process_user_input("yes")
        # Should transition to ready to save or continue workflow
        assert response2 is not None
        
        # Context 3: Test "no" responses
        agent.state_manager.context.current_state = ConversationState.INFO_EXTRACTED
        
        response3 = agent.process_user_input("no")
        # Should acknowledge decline
        assert any(keyword in response3.lower() for keyword in [
            'no problem', 'fine', 'understood', 'what would you like'
        ])


class TestFileProcessingIntegration:
    """Test file processing integration scenarios"""
    
    def test_multiple_file_types_workflow(self):
        """Test processing different file types in sequence"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            txt_file = os.path.join(temp_dir, "email.txt")
            with open(txt_file, 'w') as f:
                f.write("From: test@example.com\nSubject: Text Email\nThis is a text email.")
            
            # Mock file processing
            with patch('src.assistant.conversational_agent.EmailLLMProcessor') as mock_processor_class:
                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_processor.text = None
                mock_processor.key_info = None
                mock_processor.last_draft = None
                mock_processor.history = []
                
                agent = ConversationalEmailAgent()
                
                # Process text file
                mock_processor.load_text = Mock()
                mock_processor.extract_key_info = Mock()
                mock_processor.key_info = {'summary': 'Text email summary'}
                mock_processor.text = "Text file content"
                
                response1 = agent.process_user_input(f"Process file: {txt_file}")
                assert "processed" in response1.lower() or "loaded" in response1.lower()
                
                # Verify file was processed
                mock_processor.load_text.assert_called_once()
    
    def test_invalid_file_handling(self):
        """Test handling of invalid or non-existent files"""
        with patch('src.assistant.conversational_agent.EmailLLMProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.text = None
            mock_processor.key_info = None
            mock_processor.last_draft = None
            mock_processor.history = []
            
            # Mock file loading to raise exception
            mock_processor.load_text = Mock(side_effect=FileNotFoundError("File not found"))
            
            agent = ConversationalEmailAgent()
            
            response = agent.process_user_input("Process file: /nonexistent/file.txt")
            
            # Should handle error gracefully
            assert any(keyword in response.lower() for keyword in [
                'error', 'problem', 'trouble', 'file', 'not found'
            ])


class TestCLIAdvancedIntegration:
    """Test advanced CLI integration scenarios"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def mock_agent(self):
        with patch('src.cli.cli.get_agent') as mock_get_agent:
            mock_agent = Mock()
            mock_get_agent.return_value = mock_agent
            mock_agent.process_user_input = Mock()
            mock_agent.reset_conversation = Mock()
            mock_agent.get_greeting_message = Mock(return_value="Hello! I'm your email assistant.")
            mock_agent.get_conversation_summary = Mock(return_value={
                'conversation_state': 'greeting',
                'conversation_count': 0,
                'successful_operations': 0,
                'failed_operations': 0,
                'has_email_loaded': False,
                'has_draft': False,
                'draft_history_count': 0
            })
            yield mock_agent
    
    def test_cli_complex_workflow_simulation(self, runner, mock_agent):
        """Test complex workflow through CLI commands"""
        # Simulate loading email
        mock_agent.process_user_input.return_value = "Email loaded successfully!"
        result1 = runner.invoke(cli, ['ask', 'Here is an email: From: test@example.com...'])
        assert result1.exit_code == 0
        assert "Email loaded successfully!" in result1.output
        
        # Simulate drafting
        mock_agent.process_user_input.return_value = "Draft created: Dear sender, thank you..."
        result2 = runner.invoke(cli, ['ask', 'Draft a professional reply'])
        assert result2.exit_code == 0
        assert "Draft created" in result2.output
        
        # Simulate refinement
        mock_agent.process_user_input.return_value = "Draft refined to be more formal."
        result3 = runner.invoke(cli, ['ask', 'Make it more formal'])
        assert result3.exit_code == 0
        assert "refined" in result3.output
        
        # Simulate saving
        mock_agent.process_user_input.return_value = "Draft saved to ~/drafts/reply.txt"
        result4 = runner.invoke(cli, ['ask', 'Save this draft'])
        assert result4.exit_code == 0
        assert "saved" in result4.output
        
        # Verify all interactions occurred
        assert mock_agent.process_user_input.call_count == 4
    
    def test_cli_error_recovery(self, runner, mock_agent):
        """Test CLI error handling and recovery"""
        # First command fails
        mock_agent.process_user_input.side_effect = Exception("Service temporarily unavailable")
        result1 = runner.invoke(cli, ['ask', 'process email'])
        assert result1.exit_code == 0
        assert "Error:" in result1.output
        
        # Second command succeeds (recovery)
        mock_agent.process_user_input.side_effect = None
        mock_agent.process_user_input.return_value = "I'm back online and ready to help!"
        result2 = runner.invoke(cli, ['ask', 'are you working now?'])
        assert result2.exit_code == 0
        assert "ready to help" in result2.output
    
    def test_cli_status_tracking(self, runner, mock_agent):
        """Test status tracking through CLI"""
        # Update mock to show progression
        mock_agent.get_conversation_summary.return_value = {
            'conversation_state': 'draft_created',
            'conversation_count': 3,
            'successful_operations': 2,
            'failed_operations': 1,
            'has_email_loaded': True,
            'has_draft': True,
            'draft_history_count': 2
        }
        
        result = runner.invoke(cli, ['status'])
        assert result.exit_code == 0
        assert "draft_created" in result.output
        assert "3" in result.output  # conversation count
        assert "✅" in result.output  # email loaded
        assert "✅" in result.output  # draft available


@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance aspects of integration"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_large_conversation_history_performance(self, mock_processor_class):
        """Test performance with large conversation history"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "draft"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Simulate large conversation
        for i in range(100):
            mock_processor.refine = Mock(return_value=f"Refined draft {i}")
            mock_processor.last_draft = f"Refined draft {i}"
            
            response = agent.process_user_input(f"Refine iteration {i}")
            assert isinstance(response, str)
            assert len(response) > 0
        
        # Verify performance is acceptable
        assert len(agent.state_manager.context.conversation_history) == 200  # 100 user + 100 assistant
        assert agent.conversation_count == 100
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_memory_usage_with_multiple_sessions(self, mock_processor_class):
        """Test memory usage with multiple email sessions"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Process multiple emails to test session archiving
        for i in range(10):
            email = f"From: sender{i}@example.com\nSubject: Email {i}\nContent {i}"
            
            mock_processor.load_text = Mock()
            mock_processor.extract_key_info = Mock()
            mock_processor.key_info = {'summary': f'Email {i} summary'}
            mock_processor.text = email
            
            response = agent.process_user_input(f"Process email {i}: {email}")
            assert "processed" in response.lower() or "loaded" in response.lower()
        
        # Verify sessions are properly managed
        # Should have archived previous sessions
        assert len(agent.state_manager.context.archived_sessions) >= 9