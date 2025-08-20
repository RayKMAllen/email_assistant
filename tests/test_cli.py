import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock, patch
from src.cli.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_agent():
    """Mock the conversational agent"""
    with patch('src.cli.cli.get_agent') as mock_get_agent:
        mock_agent = MagicMock()
        mock_get_agent.return_value = mock_agent
        yield mock_agent


def test_ask_command_success(runner, mock_agent):
    """Test the ask command with successful response"""
    mock_agent.process_user_input.return_value = "I've processed your request successfully."
    
    result = runner.invoke(cli, ["ask", "Help me with this email"])
    
    assert result.exit_code == 0
    assert "I've processed your request successfully." in result.output
    mock_agent.process_user_input.assert_called_once_with("Help me with this email")


def test_ask_command_no_message(runner, mock_agent):
    """Test the ask command without a message"""
    result = runner.invoke(cli, ["ask"])
    
    assert result.exit_code == 0
    assert "Please provide a message" in result.output


def test_ask_command_exception(runner, mock_agent):
    """Test the ask command with exception"""
    mock_agent.process_user_input.side_effect = Exception("Processing error")
    
    result = runner.invoke(cli, ["ask", "test message"])
    
    assert result.exit_code == 0
    assert "⚠️ Error: Processing error" in result.output


def test_reset_command(runner, mock_agent):
    """Test the reset command"""
    mock_agent.get_greeting_message.return_value = "Hello! I'm your email assistant."
    
    result = runner.invoke(cli, ["reset"])
    
    assert result.exit_code == 0
    assert "✨ Conversation reset!" in result.output
    assert "Hello! I'm your email assistant." in result.output
    mock_agent.reset_conversation.assert_called_once()


def test_status_command(runner, mock_agent):
    """Test the status command"""
    mock_agent.get_conversation_summary.return_value = {
        'conversation_state': 'greeting',
        'conversation_count': 5,
        'successful_operations': 3,
        'failed_operations': 1,
        'has_email_loaded': True,
        'has_draft': False,
        'draft_history_count': 0
    }
    
    result = runner.invoke(cli, ["status"])
    
    assert result.exit_code == 0
    assert "📊 Conversation Status:" in result.output
    assert "Current State: greeting" in result.output
    assert "Messages Exchanged: 5" in result.output
    assert "✅" in result.output  # Email loaded indicator


def test_help_commands(runner):
    """Test the help-commands command"""
    result = runner.invoke(cli, ["help-commands"])
    
    assert result.exit_code == 0
    assert "🤖 Email Assistant Commands:" in result.output
    assert "Natural Language Commands" in result.output
    assert "CLI Commands:" in result.output


def test_chat_command(runner, mock_agent):
    """Test the chat command (should start conversational shell)"""
    # Mock input to exit immediately
    with patch('builtins.input', side_effect=['exit']):
        result = runner.invoke(cli, ["chat"])
    
    assert result.exit_code == 0
    # Should show greeting and exit message
    mock_agent.get_greeting_message.assert_called()


def test_cli_without_subcommand(runner, mock_agent):
    """Test CLI without subcommand (should start conversational shell)"""
    # Mock input to exit immediately
    with patch('builtins.input', side_effect=['exit']):
        result = runner.invoke(cli, [])
    
    assert result.exit_code == 0
    # Should show greeting
    mock_agent.get_greeting_message.assert_called()
