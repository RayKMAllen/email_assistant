import pytest
import json
from unittest.mock import patch, MagicMock
from src.assistant.llm_session import BedrockSession
from src.assistant import utils


@pytest.fixture
def session():
    with patch("assistant.llm_session.boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_runtime = MagicMock()
        mock_boto.side_effect = [mock_client, mock_runtime]
        yield BedrockSession()


def test_send_prompt_adds_to_history_and_returns_response(session):
    session.runtime.invoke_model = MagicMock()
    fake_response = {
        "body": MagicMock(
            read=MagicMock(
                return_value=json.dumps({"content": [{"text": "model output"}]}).encode(
                    "utf-8"
                )
            )
        )
    }
    session.runtime.invoke_model.return_value = fake_response
    result = session.send_prompt("hello?")
    assert result == "model output"
    assert session.history[-2]["content"] == "hello?"
    assert session.history[-1]["content"] == "model output"


def test_extract_key_info_success(monkeypatch, session):
    session.text = "email text"
    session.send_prompt = MagicMock(return_value=json.dumps({"summary": "sum"}))
    session.extract_key_info()
    assert session.key_info == {"summary": "sum"}


def test_extract_key_info_json_decode_error(monkeypatch, session):
    session.text = "email text"
    session.send_prompt = MagicMock(return_value="not json")
    with pytest.raises(Exception) as e:
        session.extract_key_info()
    assert "Failed to parse key information" in str(e.value)


def test_extract_key_info_strips_json(monkeypatch, session):
    session.text = "email text"
    # Simulate model output with 'json' in first line
    model_output = 'json\n{"summary": "sum"}\n'
    session.send_prompt = MagicMock(return_value=model_output)
    session.extract_key_info()
    assert session.key_info == {"summary": "sum"}


def test_draft_reply_sets_last_draft(session):
    session.text = "foo"
    session.send_prompt = MagicMock(return_value="drafted reply")
    result = session.draft_reply(tone="formal")
    assert result == "drafted reply"
    assert session.last_draft == "drafted reply"
    session.send_prompt.assert_called_once()
    args = session.send_prompt.call_args[0][0]
    assert "formal" in args


def test_draft_reply_no_tone(session):
    session.text = "foo"
    session.send_prompt = MagicMock(return_value="drafted reply")
    session.draft_reply()
    args = session.send_prompt.call_args[0][0]
    assert "using a" not in args


def test_refine_calls_draft_if_no_last_draft(monkeypatch, session):
    session.last_draft = None
    session.key_info = {"summary": "sum"}
    session.draft_reply = MagicMock(return_value="drafted")
    session.send_prompt = MagicMock(return_value="refined")
    result = session.refine("improve")
    assert session.last_draft == "refined"
    assert result == "refined"
    session.draft_reply.assert_called_once()


def test_refine_uses_last_draft(monkeypatch, session):
    session.last_draft = "reply"
    session.key_info = {"summary": "sum"}
    session.send_prompt = MagicMock(return_value="refined")
    result = session.refine("make shorter")
    assert session.last_draft == "refined"
    assert result == "refined"
    args = session.send_prompt.call_args[0][0]
    assert "make shorter" in args
    assert "reply" in args
    assert "sum" in args


def test_save_draft_no_last_draft(monkeypatch, session, capsys):
    session.last_draft = None
    monkeypatch.setattr(
        utils,
        "save_draft_to_file",
        lambda *a, **k: (_ for _ in ()).throw(Exception("should not call")),
    )
    session.save_draft("file.txt")
    out = capsys.readouterr().out
    assert "No draft reply to save" in out

def test_refine_with_full_history(monkeypatch, session):
    session.last_draft = "reply"
    session.key_info = {"summary": "sum"}
    session.history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "foo"},
        {"role": "assistant", "content": "bar"},
    ]
    session.send_prompt = MagicMock(return_value="refined with history")
    result = session.refine("polish", full_history=True)
    assert session.last_draft == "refined with history"
    assert result == "refined with history"
    args = session.send_prompt.call_args[0][0]
    # Should include all history, summary, last draft, and instruction
    assert "polish" in args
    assert "reply" in args
    assert "hello" in args and "hi" in args and "foo" in args and "bar" in args


def test_extract_key_info_handles_empty_response(session):
    session.text = "email text"
    session.send_prompt = MagicMock(return_value="")
    with pytest.raises(Exception) as e:
        session.extract_key_info()
    assert "Failed to parse key information" in str(e.value)


# --- Tests for utils.py ---


def test_process_path_or_email_reads_file(tmp_path):
    file = tmp_path / "mail.txt"
    file.write_text("hello")
    result = utils.process_path_or_email(str(file))
    assert result == "hello"


def test_process_path_or_email_returns_text():
    text = "This is not a file path"
    result = utils.process_path_or_email(text)
    assert result == text


def test_save_draft_to_file(tmp_path):
    file = tmp_path / "out.txt"
    utils.save_draft_to_file("draft", str(file))
    assert file.read_text() == "draft"
