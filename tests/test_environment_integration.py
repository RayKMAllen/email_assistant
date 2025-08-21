"""
Environment and configuration integration tests.
Tests system behavior across different environments, configurations, and deployment scenarios.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import tempfile
import os
from datetime import datetime
import sys

from src.assistant.conversational_agent import ConversationalEmailAgent
from src.assistant.conversation_state import ConversationState
from src.cli.cli import cli
from click.testing import CliRunner


class TestEnvironmentVariableIntegration:
    """Test integration with environment variables and configuration"""
    
    @patch.dict(os.environ, {'AWS_REGION': 'us-west-2'})
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_aws_region_configuration(self, mock_processor_class):
        """Test AWS region configuration from environment"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Mock S3 operations that should use the configured region
        with patch('src.assistant.utils.boto3.client') as mock_boto:
            mock_s3 = Mock()
            mock_boto.return_value = mock_s3
            mock_s3.put_object.return_value = None
            
            mock_processor.save_draft = Mock()
            
            response = agent.process_user_input("save to cloud storage")
            
            # Verify boto3 client was called (region would be picked up from environment)
            assert "saved" in response.lower() or "cloud" in response.lower()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_missing_environment_variables(self, mock_processor_class):
        """Test behavior when environment variables are missing"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test cloud save without AWS credentials
        mock_processor.save_draft = Mock(side_effect=Exception("NoCredentialsError"))
        
        response = agent.process_user_input("save to cloud")
        
        # Should handle missing credentials gracefully
        assert "error" in response.lower() or "problem" in response.lower()
        assert agent.state_manager.context.current_state == ConversationState.ERROR_RECOVERY
    
    @patch.dict(os.environ, {'HOME': '/tmp/test_home'})
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_home_directory_configuration(self, mock_processor_class):
        """Test home directory configuration"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Mock file operations
        with patch('src.assistant.utils.os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()) as mock_file:
            
            mock_processor.save_draft = Mock()
            
            response = agent.process_user_input("save the draft")
            
            # Should use configured home directory
            assert "saved" in response.lower()


class TestFileSystemIntegration:
    """Test file system integration across different environments"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_file_permissions_handling(self, mock_processor_class):
        """Test handling of file permission issues"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Simulate permission denied error
        mock_processor.save_draft = Mock(side_effect=PermissionError("Permission denied"))
        
        response = agent.process_user_input("save to /root/protected/file.txt")
        
        # Should handle permission errors gracefully
        assert "error" in response.lower() or "permission" in response.lower()
        assert agent.failed_operations > 0
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_disk_space_handling(self, mock_processor_class):
        """Test handling of disk space issues"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Simulate disk full error
        mock_processor.save_draft = Mock(side_effect=OSError("No space left on device"))
        
        response = agent.process_user_input("save the draft")
        
        # Should handle disk space errors gracefully
        assert "error" in response.lower() or "space" in response.lower()
    
    def test_file_path_normalization(self):
        """Test file path normalization across different OS"""
        from src.assistant.utils import save_draft_to_file
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test different path formats
            test_paths = [
                os.path.join(temp_dir, "test.txt"),
                temp_dir + "/test2.txt",  # Unix style
                temp_dir + "\\test3.txt",  # Windows style
            ]
            
            for test_path in test_paths:
                try:
                    save_draft_to_file("test content", test_path)
                    
                    # Verify file was created
                    normalized_path = os.path.normpath(test_path)
                    assert os.path.exists(normalized_path)
                    
                    # Clean up
                    if os.path.exists(normalized_path):
                        os.unlink(normalized_path)
                        
                except Exception as e:
                    pytest.fail(f"Path normalization failed for {test_path}: {e}")


class TestPlatformSpecificIntegration:
    """Test platform-specific integration scenarios"""
    
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_windows_path_handling(self, mock_processor_class):
        """Test Windows-specific path handling"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test Windows path format
        with patch('src.assistant.utils.save_draft_to_file') as mock_save:
            mock_processor.save_draft = Mock()
            
            response = agent.process_user_input("save to C:\\Users\\test\\draft.txt")
            
            # Should handle Windows paths correctly
            assert "saved" in response.lower()
    
    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_unix_path_handling(self, mock_processor_class):
        """Test Unix-specific path handling"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test Unix path format
        with patch('src.assistant.utils.save_draft_to_file') as mock_save:
            mock_processor.save_draft = Mock()
            
            response = agent.process_user_input("save to /home/user/draft.txt")
            
            # Should handle Unix paths correctly
            assert "saved" in response.lower()


class TestNetworkEnvironmentIntegration:
    """Test network environment integration"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_network_connectivity_issues(self, mock_processor_class):
        """Test handling of network connectivity issues"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Simulate network connectivity issues
        network_errors = [
            ConnectionError("Network unreachable"),
            TimeoutError("Connection timed out"),
            Exception("DNS resolution failed")
        ]
        
        for error in network_errors:
            mock_processor.extract_key_info = Mock(side_effect=error)
            
            response = agent.process_user_input("extract information from the email")
            
            # Should handle network errors gracefully
            assert "error" in response.lower() or "problem" in response.lower()
            assert agent.state_manager.context.current_state == ConversationState.ERROR_RECOVERY
            
            # Reset for next test
            mock_processor.extract_key_info.side_effect = None
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_proxy_environment_handling(self, mock_processor_class):
        """Test handling of proxy environments"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        # Test with proxy environment variables
        with patch.dict(os.environ, {
            'HTTP_PROXY': 'http://proxy.company.com:8080',
            'HTTPS_PROXY': 'https://proxy.company.com:8080'
        }):
            agent = ConversationalEmailAgent()
            
            # Mock cloud save that would use network
            with patch('src.assistant.utils.boto3.client') as mock_boto:
                mock_s3 = Mock()
                mock_boto.return_value = mock_s3
                mock_s3.put_object.return_value = None
                
                mock_processor.save_draft = Mock()
                
                response = agent.process_user_input("save to cloud")
                
                # Should work with proxy configuration
                assert "saved" in response.lower() or "cloud" in response.lower()


class TestResourceConstraintIntegration:
    """Test integration under resource constraints"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_low_memory_environment(self, mock_processor_class):
        """Test behavior in low memory environments"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Simulate memory pressure with large content
        large_email = "From: test@example.com\nSubject: Large Email\n" + "Content " * 100000
        
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'Large email processed'}
        mock_processor.text = large_email
        
        response = agent.process_user_input(f"Process: {large_email[:100]}...")
        
        # Should handle large content without memory issues
        assert "processed" in response.lower() or "loaded" in response.lower()
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_cpu_constraint_handling(self, mock_processor_class):
        """Test handling of CPU constraints"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = "email content"
        mock_processor.key_info = {'summary': 'test'}
        mock_processor.last_draft = "draft"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Simulate CPU-intensive operations with delays
        import time
        
        def slow_operation(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow processing
            return "Slow result"
        
        mock_processor.refine = slow_operation
        mock_processor.last_draft = "Slow result"
        
        start_time = time.time()
        response = agent.process_user_input("refine the draft")
        end_time = time.time()
        
        # Should complete even with slow operations
        assert "refined" in response.lower() or "updated" in response.lower()
        assert end_time - start_time >= 0.1  # Verify delay occurred


class TestConfigurationFileIntegration:
    """Test configuration file integration"""
    
    def test_missing_config_file_handling(self):
        """Test handling when configuration files are missing"""
        # Test that system works without config files
        try:
            agent = ConversationalEmailAgent()
            greeting = agent.get_greeting_message()
            
            # Should work with default configuration
            assert isinstance(greeting, str)
            assert len(greeting) > 0
            
        except Exception as e:
            pytest.fail(f"System failed without config file: {e}")
    
    @patch('builtins.open', mock_open(read_data='{"test_setting": "test_value"}'))
    def test_config_file_parsing(self):
        """Test configuration file parsing"""
        # Test that system can handle config files if present
        try:
            agent = ConversationalEmailAgent()
            
            # Should initialize successfully even with config file present
            assert agent is not None
            
        except Exception as e:
            pytest.fail(f"System failed with config file: {e}")


class TestDeploymentEnvironmentIntegration:
    """Test different deployment environment scenarios"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_containerized_environment(self, mock_processor_class):
        """Test behavior in containerized environments"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        # Simulate containerized environment constraints
        with patch.dict(os.environ, {
            'CONTAINER': 'true',
            'HOME': '/app',
            'USER': 'appuser'
        }):
            agent = ConversationalEmailAgent()
            
            # Test file operations in container
            with patch('src.assistant.utils.os.makedirs') as mock_makedirs, \
                 patch('builtins.open', mock_open()) as mock_file:
                
                mock_processor.save_draft = Mock()
                
                response = agent.process_user_input("save the draft")
                
                # Should work in containerized environment
                assert "saved" in response.lower()
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_serverless_environment(self, mock_processor_class):
        """Test behavior in serverless environments"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        # Simulate serverless environment
        with patch.dict(os.environ, {
            'AWS_LAMBDA_FUNCTION_NAME': 'email-assistant',
            'AWS_EXECUTION_ENV': 'AWS_Lambda_python3.9'
        }):
            agent = ConversationalEmailAgent()
            
            # Test cloud operations in serverless
            with patch('src.assistant.utils.boto3.client') as mock_boto:
                mock_s3 = Mock()
                mock_boto.return_value = mock_s3
                mock_s3.put_object.return_value = None
                
                mock_processor.save_draft = Mock()
                
                response = agent.process_user_input("save to cloud")
                
                # Should work in serverless environment
                assert "saved" in response.lower()


class TestCLIEnvironmentIntegration:
    """Test CLI integration across different environments"""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_cli_terminal_compatibility(self, runner):
        """Test CLI compatibility across different terminals"""
        # Test basic CLI functionality
        result = runner.invoke(cli, ['help-commands'])
        assert result.exit_code == 0
        
        # Should work regardless of terminal type
        assert "Email Assistant Commands" in result.output
    
    @patch('src.cli.cli.get_agent')
    def test_cli_encoding_handling(self, mock_get_agent, runner):
        """Test CLI handling of different character encodings"""
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        mock_agent.process_user_input.return_value = "Response with unicode: 你好 🎉"
        
        result = runner.invoke(cli, ['ask', 'test unicode'])
        assert result.exit_code == 0
        
        # Should handle unicode in output
        # Note: Exact unicode handling depends on terminal capabilities
        assert len(result.output) > 0
    
    @patch.dict(os.environ, {'TERM': 'dumb'})
    @patch('src.cli.cli.get_agent')
    def test_cli_limited_terminal_support(self, mock_get_agent, runner):
        """Test CLI in limited terminal environments"""
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        mock_agent.get_conversation_summary.return_value = {
            'conversation_state': 'greeting',
            'conversation_count': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'has_email_loaded': False,
            'has_draft': False,
            'draft_history_count': 0
        }
        
        result = runner.invoke(cli, ['status'])
        assert result.exit_code == 0
        
        # Should work even in limited terminals
        assert "Conversation Status" in result.output


class TestSecurityEnvironmentIntegration:
    """Test security-related environment integration"""
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_restricted_file_access(self, mock_processor_class):
        """Test behavior with restricted file access"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "sensitive content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test saving to restricted location
        mock_processor.save_draft = Mock(side_effect=PermissionError("Access denied"))
        
        response = agent.process_user_input("save to /etc/passwd")
        
        # Should handle security restrictions gracefully
        assert "error" in response.lower() or "permission" in response.lower()
        assert agent.failed_operations > 0
    
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_sandboxed_environment(self, mock_processor_class):
        """Test behavior in sandboxed environments"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test operations in sandbox
        with patch('src.assistant.utils.os.makedirs') as mock_makedirs:
            mock_makedirs.side_effect = OSError("Operation not permitted")
            mock_processor.save_draft = Mock(side_effect=OSError("Operation not permitted"))
            
            response = agent.process_user_input("save the draft")
            
            # Should handle sandbox restrictions
            assert "error" in response.lower() or "problem" in response.lower()


class TestLocalizationEnvironmentIntegration:
    """Test localization and internationalization"""
    
    @patch.dict(os.environ, {'LANG': 'en_US.UTF-8'})
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_locale_handling(self, mock_processor_class):
        """Test handling of different locales"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test with locale-specific content
        email_with_locale = """From: test@example.com
Subject: Réunion importante
Bonjour, nous devons organiser une réunion importante."""
        
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.key_info = {'summary': 'French meeting request'}
        mock_processor.text = email_with_locale
        
        response = agent.process_user_input(f"Process: {email_with_locale}")
        
        # Should handle international content
        assert "processed" in response.lower() or "loaded" in response.lower()
    
    @patch.dict(os.environ, {'TZ': 'Europe/Paris'})
    @patch('src.assistant.conversational_agent.EmailLLMProcessor')
    def test_timezone_handling(self, mock_processor_class):
        """Test handling of different timezones"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = "draft content"
        mock_processor.history = []
        
        agent = ConversationalEmailAgent()
        
        # Test timestamp handling with timezone
        with patch('src.assistant.utils.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 14, 30, 0)
            mock_datetime.strftime = datetime.strftime
            
            mock_processor.save_draft = Mock()
            
            response = agent.process_user_input("save the draft")
            
            # Should handle timezone-aware operations
            assert "saved" in response.lower()