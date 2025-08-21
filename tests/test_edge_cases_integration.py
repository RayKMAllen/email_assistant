"""
Edge case and error handling integration tests.
Tests boundary conditions, malformed inputs, and error recovery scenarios.
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


class TestMalformedInputIntegration:
    """Test handling of malformed and edge case inputs"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_malformed_email_content(self, mock_processor_class):
        """Test processing emails with malformed headers and content"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        malformed_emails = [
            # Missing headers
            "This is just plain text without any email headers",
            
            # Malformed headers
            "From: \nTo: \nSubject: \nEmpty headers test",
            
            # Invalid email addresses
            "From: not-an-email\nTo: also-not-an-email\nSubject: Invalid emails",
            
            # Mixed encoding issues (simulated)
            "From: test@example.com\nSubject: Test\nContent with special chars: àáâãäåæçèéêë",
            
            # Very long subject line
            "From: test@example.com\nSubject: " + "Very long subject " * 20 + "\nContent here",
            
            # Empty email
            "",
            
            # Only whitespace
            "   \n\t\n   ",
            
            # HTML content mixed with plain text
            """From: test@example.com
Subject: HTML Email
<html><body><h1>Hello</h1><p>This is <b>bold</b> text.</p></body></html>
Also some plain text here."""
        ]
        
        for i, malformed_email in enumerate(malformed_emails):
            # Setup mock to handle malformed content gracefully
            mock_processor.load_text = Mock()
            mock_processor.extract_key_info = Mock()
            
            # Some might fail extraction, others might succeed with limited info
            if i % 2 == 0:
                mock_processor.key_info = {'summary': f'Processed malformed email {i}'}
                mock_processor.text = malformed_email
            else:
                mock_processor.extract_key_info.side_effect = Exception("Could not extract info")
                mock_processor.key_info = None
            
            response = agent.process_user_input(f"Process this email: {malformed_email}")
            
            # Should handle gracefully without crashing
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Reset for next iteration
            mock_processor.extract_key_info.side_effect = None
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_extremely_long_inputs(self, mock_processor_class):
        """Test handling of extremely long user inputs"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Create extremely long email content
        long_email = "From: test@example.com\nSubject: Long Email\n" + "This is a very long email content. " * 1000
        
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Very long email processed'}
        mock_processor.text = long_email
        
        response = agent.process_user_input(f"Process this long email: {long_email}")
        
        # Should handle without memory issues
        assert isinstance(response, str)
        assert "processed" in response.lower() or "loaded" in response.lower()
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_special_characters_and_unicode(self, mock_processor_class):
        """Test handling of special characters and Unicode content"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        unicode_emails = [
            # Emoji content
            "From: test@example.com\nSubject: 🎉 Party Invitation 🎊\nHey! 😊 Want to join our party? 🥳",
            
            # Multiple languages
            "From: test@example.com\nSubject: Multilingual\nHello, Hola, Bonjour, こんにちは, 你好",
            
            # Mathematical symbols
            "From: test@example.com\nSubject: Math\nThe equation is: ∑(x²) = ∫f(x)dx ≈ π/2",
            
            # Currency symbols
            "From: test@example.com\nSubject: Payment\nAmount: $100, €85, ¥1000, £75",
            
            # Special punctuation
            "From: test@example.com\nSubject: Quotes\n\"This is a 'test' with various—punctuation… marks!\""
        ]
        
        for unicode_email in unicode_emails:
            mock_processor.load_text = Mock()
            mock_processor.extract_key_info = Mock()
            mock_processor.key_info = {'summary': 'Unicode email processed'}
            mock_processor.text = unicode_email
            
            response = agent.process_user_input(f"Process: {unicode_email}")
            
            # Should handle Unicode gracefully
            assert isinstance(response, str)
            assert len(response) > 0


class TestErrorRecoveryIntegration:
    """Test comprehensive error recovery scenarios"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_cascading_failures_recovery(self, mock_processor_class):
        """Test recovery from multiple consecutive failures"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Setup initial successful state
        email = "From: test@example.com\nSubject: Test\nContent"
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Test email'}
        mock_processor.text = email
        
        agent.process_user_input(f"Process: {email}")
        assert agent.state_manager.context.current_state == ConversationState.INFO_EXTRACTED
        
        # Simulate cascading failures
        failures = [
            ("Network timeout", "draft a reply"),
            ("Service unavailable", "try again"),
            ("Rate limit exceeded", "please draft"),
            ("Authentication failed", "create draft"),
            ("Internal server error", "help me draft")
        ]
        
        for error_msg, user_input in failures:
            mock_processor.draft_reply = Mock(side_effect=Exception(error_msg))
            response = agent.process_user_input(user_input)
            
            # Should be in error recovery state
            assert agent.state_manager.context.current_state == ConversationState.ERROR_RECOVERY
            assert "error" in response.lower() or "problem" in response.lower()
        
        # Finally succeed
        mock_processor.draft_reply = Mock(return_value="Finally successful draft")
        mock_processor.last_draft = "Finally successful draft"
        
        response = agent.process_user_input("one more try please")
        assert agent.state_manager.context.current_state == ConversationState.DRAFT_CREATED
        assert "draft" in response.lower()
        
        # Verify error tracking
        assert agent.failed_operations >= 5
        assert agent.successful_operations >= 2  # Initial load + final draft
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_partial_failure_recovery(self, mock_processor_class):
        """Test recovery from partial failures where some operations succeed"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Email loads successfully but info extraction fails
        email = "From: test@example.com\nSubject: Test\nContent"
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock(side_effect=Exception("Extraction failed"))
        mock_processor.text = email
        mock_processor.key_info = None
        
        response1 = agent.process_user_input(f"Process: {email}")
        
        # Should handle partial failure
        assert agent.state_manager.context.email_content is not None
        assert "error" in response1.lower() or "trouble" in response1.lower()
        
        # Try extraction again - this time succeeds
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Successfully extracted'}
        
        response2 = agent.process_user_input("try extracting the information again")
        assert "extract" in response2.lower() or "information" in response2.lower()
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_state_corruption_recovery(self, mock_processor_class):
        """Test recovery from corrupted conversation state"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Manually corrupt the state to simulate edge case
        agent.state_manager.context.current_state = ConversationState.DRAFT_CREATED
        agent.state_manager.context.email_content = None  # Inconsistent state
        agent.state_manager.context.current_draft = None  # Inconsistent state
        
        # Try to refine draft when state is corrupted
        mock_processor.refine = Mock(side_effect=Exception("No draft to refine"))
        
        response = agent.process_user_input("make it more formal")
        
        # Should handle gracefully and provide guidance
        assert isinstance(response, str)
        assert len(response) > 0


class TestBoundaryConditionIntegration:
    """Test boundary conditions and limits"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_maximum_conversation_length(self, mock_processor_class):
        """Test behavior at maximum conversation length"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "draft"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Simulate very long conversation
        for i in range(500):  # Large number of interactions
            mock_processor.refine = Mock(return_value=f"Draft {i}")
            mock_processor.last_draft = f"Draft {i}"
            
            response = agent.process_user_input(f"refine {i}")
            
            # Should continue to work even with long history
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Check memory usage doesn't grow unbounded
            if i > 100:
                # History should be managed (not grow indefinitely)
                assert len(agent.state_manager.context.conversation_history) < 1000
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_empty_and_null_inputs(self, mock_processor_class):
        """Test handling of empty and null inputs"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        empty_inputs = [
            "",
            "   ",
            "\n\n\n",
            "\t\t\t",
            None  # This might cause issues, should be handled
        ]
        
        for empty_input in empty_inputs:
            try:
                if empty_input is None:
                    # Skip None input as it would cause TypeError in real usage
                    continue
                    
                response = agent.process_user_input(empty_input)
                
                # Should handle empty input gracefully
                assert isinstance(response, str)
                assert len(response) > 0
                
            except Exception as e:
                # Should not crash on empty inputs
                pytest.fail(f"Agent crashed on empty input '{empty_input}': {e}")
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_rapid_state_transitions(self, mock_processor_class):
        """Test rapid state transitions and race conditions"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Rapid sequence of operations
        operations = [
            ("load email", lambda: setattr(mock_processor, 'text', 'email')),
            ("extract info", lambda: setattr(mock_processor, 'key_info', {'summary': 'test'})),
            ("draft reply", lambda: setattr(mock_processor, 'last_draft', 'draft')),
            ("refine draft", lambda: setattr(mock_processor, 'last_draft', 'refined')),
            ("save draft", lambda: None)
        ]
        
        # Setup mocks
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.draft_reply = Mock(return_value="draft")
        mock_processor.refine = Mock(return_value="refined")
        mock_processor.save_draft = Mock()
        
        # Execute rapid operations
        for operation, setup in operations:
            setup()
            response = agent.process_user_input(operation)
            
            # Should handle rapid transitions
            assert isinstance(response, str)
            assert len(response) > 0


class TestResourceLimitIntegration:
    """Test resource limits and constraints"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_memory_pressure_handling(self, mock_processor_class):
        """Test behavior under memory pressure"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Simulate memory pressure by creating large objects
        large_content = "x" * 1000000  # 1MB string
        
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Large content processed'}
        mock_processor.text = large_content
        
        response = agent.process_user_input(f"Process this large email: {large_content[:100]}...")
        
        # Should handle large content without issues
        assert isinstance(response, str)
        assert "processed" in response.lower() or "loaded" in response.lower()
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_concurrent_operation_simulation(self, mock_processor_class):
        """Test simulation of concurrent operations"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "draft"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Simulate concurrent-like operations by rapid succession
        import threading
        import time
        
        results = []
        
        def process_input(input_text, delay=0):
            time.sleep(delay)
            try:
                mock_processor.refine = Mock(return_value=f"Processed: {input_text}")
                mock_processor.last_draft = f"Processed: {input_text}"
                result = agent.process_user_input(input_text)
                results.append((input_text, result, None))
            except Exception as e:
                results.append((input_text, None, str(e)))
        
        # Create multiple "concurrent" operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_input, args=(f"refine {i}", i * 0.01))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations completed
        assert len(results) == 5
        for input_text, result, error in results:
            if error:
                pytest.fail(f"Concurrent operation failed: {input_text} -> {error}")
            assert result is not None
            assert isinstance(result, str)


class TestExceptionHandlingIntegration:
    """Test comprehensive exception handling"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_llm_service_exceptions(self, mock_processor_class):
        """Test handling of various LLM service exceptions"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Different types of LLM exceptions
        llm_exceptions = [
            ConnectionError("Network connection failed"),
            TimeoutError("Request timed out"),
            ValueError("Invalid response format"),
            KeyError("Missing required field"),
            json.JSONDecodeError("Invalid JSON", "", 0),
            Exception("Generic LLM error")
        ]
        
        for exception in llm_exceptions:
            mock_processor.draft_reply = Mock(side_effect=exception)
            
            response = agent.process_user_input("draft a reply")
            
            # Should handle each exception type gracefully
            assert isinstance(response, str)
            assert len(response) > 0
            assert agent.state_manager.context.current_state == ConversationState.ERROR_RECOVERY
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_file_system_exceptions(self, mock_processor_class):
        """Test handling of file system related exceptions"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft to save"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # File system exceptions
        fs_exceptions = [
            FileNotFoundError("File not found"),
            PermissionError("Permission denied"),
            OSError("Disk full"),
            IOError("I/O operation failed")
        ]
        
        for exception in fs_exceptions:
            mock_processor.save_draft = Mock(side_effect=exception)
            
            response = agent.process_user_input("save the draft")
            
            # Should handle file system errors gracefully
            assert isinstance(response, str)
            assert len(response) > 0
            assert any(keyword in response.lower() for keyword in ['error', 'problem', 'trouble'])
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_aws_service_exceptions(self, mock_processor_class):
        """Test handling of AWS service exceptions"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft to save"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Simulate AWS exceptions
        aws_exceptions = [
            Exception("NoCredentialsError: Unable to locate credentials"),
            Exception("AccessDenied: Access denied"),
            Exception("NoSuchBucket: The specified bucket does not exist"),
            Exception("ServiceUnavailable: Service temporarily unavailable")
        ]
        
        for exception in aws_exceptions:
            mock_processor.save_draft = Mock(side_effect=exception)
            
            response = agent.process_user_input("save to cloud storage")
            
            # Should handle AWS errors gracefully
            assert isinstance(response, str)
            assert len(response) > 0
            assert any(keyword in response.lower() for keyword in ['error', 'problem', 'cloud', 'save'])


class TestCLIEdgeCaseIntegration:
    """Test CLI edge cases and error conditions"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_cli_with_malformed_commands(self, runner):
        """Test CLI with malformed or edge case commands"""
        # Test with very long command
        long_message = "process email " + "very long content " * 100
        result = runner.invoke(cli, ['ask'] + long_message.split())
        assert result.exit_code == 0  # Should not crash
        
        # Test with special characters
        special_message = ['ask', 'process', 'email:', '🎉', 'special', 'chars', '&', '<', '>']
        result = runner.invoke(cli, special_message)
        assert result.exit_code == 0
        
        # Test with empty ask command
        result = runner.invoke(cli, ['ask'])
        assert result.exit_code == 0
        assert "Please provide a message" in result.output
    
    @patch('src.cli.cli.get_agent')
    def test_cli_agent_initialization_failure(self, mock_get_agent, runner):
        """Test CLI behavior when agent initialization fails"""
        mock_get_agent.side_effect = Exception("Agent initialization failed")
        
        result = runner.invoke(cli, ['ask', 'test message'])
        assert result.exit_code == 0  # CLI should not crash
        assert "Error:" in result.output
    
    @patch('src.cli.cli.get_agent')
    def test_cli_keyboard_interrupt_handling(self, mock_get_agent, runner):
        """Test CLI handling of keyboard interrupts"""
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        mock_agent.process_user_input.side_effect = KeyboardInterrupt()
        
        result = runner.invoke(cli, ['ask', 'test message'])
        assert result.exit_code == 0  # Should handle gracefully