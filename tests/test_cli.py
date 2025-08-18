import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from src.cli.cli import cli

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def mock_session(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("src.cli.cli.session", mock)
    return mock

def test_load_success_text(runner, mock_session):
    mock_session.load_text.return_value = None
    mock_session.extract_key_info.return_value = None
    result = runner.invoke(cli, ["load", "This is an email"])
    assert "Email content loaded successfully." in result.output
    assert "Type 'draft' to draft a reply" in result.output
    assert result.exit_code == 0
    mock_session.load_text.assert_called_once()
    mock_session.extract_key_info.assert_called_once()

def test_load_failure_load_text(runner, mock_session):
    mock_session.load_text.side_effect = Exception("fail load")
    result = runner.invoke(cli, ["load", "bad input"])
    assert "⚠️ Error loading email: fail load" in result.output
    assert result.exit_code == 0

def test_load_failure_extract_key_info(runner, mock_session):
    mock_session.load_text.return_value = None
    mock_session.extract_key_info.side_effect = Exception("fail extract")
    result = runner.invoke(cli, ["load", "good input"])
    assert "Email content loaded successfully." in result.output
    assert "⚠️ Error extracting key info: fail extract" in result.output

def test_draft_success(runner, mock_session):
    mock_session.text = "email"
    mock_session.key_info = "info"
    mock_session.draft_reply.return_value = "Drafted reply text"
    result = runner.invoke(cli, ["draft", "formal"])
    assert "Drafted reply:" in result.output
    assert "Drafted reply text" in result.output
    assert "Type 'save' to save the draft" in result.output
    assert result.exit_code == 0
    mock_session.draft_reply.assert_called_once_with(tone="formal")

def test_draft_no_email_loaded(runner, mock_session):
    mock_session.text = None
    mock_session.key_info = None
    result = runner.invoke(cli, ["draft"])
    assert "⚠️ Correct email information has not yet been loaded" in result.output
    assert result.exit_code == 0

def test_draft_exception(runner, mock_session):
    mock_session.text = "email"
    mock_session.key_info = "info"
    mock_session.draft_reply.side_effect = Exception("draft error")
    result = runner.invoke(cli, ["draft"])
    assert "⚠️ Error drafting reply: draft error" in result.output

def test_save_success_with_filepath(runner, mock_session):
    mock_session.last_draft = "reply"
    result = runner.invoke(cli, ["save", "output.txt"])
    assert "Draft saved successfully." in result.output
    mock_session.save_draft.assert_called_once_with("output.txt")

def test_save_success_default(runner, mock_session):
    mock_session.last_draft = "reply"
    result = runner.invoke(cli, ["save"])
    assert "Draft saved to default location." in result.output
    mock_session.save_draft.assert_called_once_with(None)

def test_save_no_draft(runner, mock_session):
    mock_session.last_draft = None
    result = runner.invoke(cli, ["save"])
    assert "⚠️ No draft available. Please use the 'draft' command first." in result.output

def test_save_exception(runner, mock_session):
    mock_session.last_draft = "reply"
    mock_session.save_draft.side_effect = Exception("save error")
    result = runner.invoke(cli, ["save"])
    assert "⚠️ Error saving draft: save error" in result.output

def test_refine_success(runner, mock_session):
    mock_session.last_draft = "reply"
    mock_session.refine.return_value = "Refined reply"
    result = runner.invoke(cli, ["refine", "make", "it", "shorter"])
    assert "Refined reply:" in result.output
    assert "Refined reply" in result.output
    assert "Type 'save' to save the refined draft" in result.output
    mock_session.refine.assert_called_once_with("make it shorter")

def test_refine_no_draft(runner, mock_session):
    mock_session.last_draft = None
    result = runner.invoke(cli, ["refine", "improve"])
    assert "⚠️ No draft available. Please use the 'draft' command first." in result.output

def test_refine_exception(runner, mock_session):
    mock_session.last_draft = "reply"
    mock_session.refine.side_effect = Exception("refine error")
    result = runner.invoke(cli, ["refine", "improve"])
    assert "⚠️ Error refining reply: refine error" in result.output

def test_exit_command(runner):
    result = runner.invoke(cli, ["exit"])
    assert "👋 Goodbye!" in result.output
    assert result.exit_code == 0