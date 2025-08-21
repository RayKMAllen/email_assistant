"""
Pytest configuration and shared fixtures for the email assistant test suite.
"""

import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
import json

from assistant.conversation_state import ConversationContext, ConversationState


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_email():
    """Sample email content for testing"""
    return """From: john.doe@example.com
To: jane.smith@company.com
Subject: Project Update Meeting

Dear Jane,

I hope this email finds you well. I wanted to schedule a meeting to discuss the current status of our project.

Would next Tuesday at 2 PM work for you? We can meet in the conference room or via video call.

Please let me know your availability.

Best regards,
John Doe
Project Manager
john.doe@example.com
(555) 123-4567"""


@pytest.fixture
def sample_email_extracted_info():
    """Sample extracted information from email"""
    return {
        "summary": "Meeting request from John Doe to discuss project status",
        "sender_name": "John Doe",
        "sender_email": "john.doe@example.com",
        "receiver_name": "Jane Smith",
        "receiver_email": "jane.smith@company.com",
        "subject": "Project Update Meeting",
        "sender_contact_details": {
            "email": "john.doe@example.com",
            "phone": "(555) 123-4567",
            "title": "Project Manager"
        },
        "key_points": [
            "Schedule meeting for project status discussion",
            "Proposed time: Tuesday at 2 PM",
            "Meeting can be in-person or video call"
        ],
        "action_required": "Respond with availability confirmation"
    }


@pytest.fixture
def sample_draft_reply():
    """Sample draft reply"""
    return """Dear John,

Thank you for your email regarding the project update meeting.

Tuesday at 2 PM works perfectly for me. I would prefer to meet via video call if that's convenient for you.

Please send me the meeting invitation with the video call details.

Looking forward to discussing the project status with you.

Best regards,
Jane Smith"""


@pytest.fixture
def mock_bedrock_response():
    """Mock Bedrock API response"""
    def _create_response(content):
        return {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps({"content": [{"text": content}]}).encode("utf-8")
                )
            )
        }
    return _create_response


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing"""
    with patch('boto3.client') as mock_boto:
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.head_bucket.return_value = None
        mock_s3.put_object.return_value = None
        yield mock_s3


@pytest.fixture
def conversation_context():
    """Create a conversation context for testing"""
    context = ConversationContext()
    context.current_state = ConversationState.GREETING
    return context


@pytest.fixture
def populated_conversation_context(sample_email, sample_email_extracted_info, sample_draft_reply):
    """Create a populated conversation context"""
    context = ConversationContext()
    context.current_state = ConversationState.DRAFT_CREATED
    context.email_content = sample_email
    context.extracted_info = sample_email_extracted_info
    context.current_draft = sample_draft_reply
    context.draft_history = [sample_draft_reply]
    
    # Add some conversation history
    context.add_to_history("user", "Here's an email I need help with")
    context.add_to_history("assistant", "I've processed your email successfully")
    context.add_to_history("user", "Draft a professional reply")
    context.add_to_history("assistant", "I've created a draft reply for you")
    
    return context


@pytest.fixture
def mock_llm_processor():
    """Mock EmailLLMProcessor for testing"""
    with patch('src.assistant.llm_session.EmailLLMProcessor') as mock_class:
        mock_processor = Mock()
        mock_class.return_value = mock_processor
        
        # Set default attributes
        mock_processor.text = None
        mock_processor.key_info = None
        mock_processor.last_draft = None
        mock_processor.history = []
        mock_processor.s3_client = Mock()
        mock_processor.runtime = Mock()
        
        # Set default method behaviors
        mock_processor.load_text = Mock()
        mock_processor.extract_key_info = Mock()
        mock_processor.draft_reply = Mock()
        mock_processor.refine = Mock()
        mock_processor.save_draft = Mock()
        mock_processor.send_prompt = Mock()
        
        yield mock_processor


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing"""
    with patch('src.assistant.utils.os.path.isfile') as mock_isfile, \
         patch('src.assistant.utils.extract_text') as mock_extract, \
         patch('builtins.open', create=True) as mock_open:
        
        mock_isfile.return_value = False  # Default to treating input as text
        mock_extract.return_value = "extracted file content"
        
        yield {
            'isfile': mock_isfile,
            'extract_text': mock_extract,
            'open': mock_open
        }


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed for consistent test results"""
    import random
    random.seed(42)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "aws: mark test as requiring AWS credentials"
    )
    config.addinivalue_line(
        "markers", "llm: mark test as requiring LLM access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ['integration', 'slow', 'aws', 'llm'] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to performance tests
        if 'performance' in item.name.lower() or 'large' in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add aws marker to S3/cloud tests
        if any(keyword in item.name.lower() for keyword in ['s3', 'cloud', 'aws']):
            item.add_marker(pytest.mark.aws)
        
        # Add llm marker to tests that use LLM
        if any(keyword in item.name.lower() for keyword in ['llm', 'bedrock', 'prompt']):
            item.add_marker(pytest.mark.llm)


# Custom assertions
def assert_valid_email_format(email_content):
    """Assert that content looks like a valid email"""
    assert isinstance(email_content, str)
    assert len(email_content.strip()) > 0
    # Should have some email-like characteristics
    email_indicators = ['dear', 'hi', 'hello', 'regards', 'sincerely', 'best', 'thank you']
    content_lower = email_content.lower()
    assert any(indicator in content_lower for indicator in email_indicators), \
        f"Content doesn't look like an email: {email_content[:100]}..."


def assert_valid_conversation_state(state):
    """Assert that state is a valid conversation state"""
    assert isinstance(state, ConversationState)
    assert state in ConversationState


def assert_successful_response(response):
    """Assert that response indicates success"""
    assert isinstance(response, str)
    assert len(response.strip()) > 0
    # Should not contain obvious error indicators
    error_indicators = ['error', 'failed', 'exception', 'traceback']
    response_lower = response.lower()
    assert not any(indicator in response_lower for indicator in error_indicators), \
        f"Response appears to contain errors: {response}"


# Test data generators
class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def generate_email(sender="test@example.com", receiver="user@example.com", 
                      subject="Test Email", content="Test email content"):
        """Generate email content"""
        return f"""From: {sender}
To: {receiver}
Subject: {subject}

{content}

Best regards,
Test User"""
    
    @staticmethod
    def generate_extracted_info(summary="Test summary", sender="Test User"):
        """Generate extracted email information"""
        return {
            "summary": summary,
            "sender_name": sender,
            "sender_email": "test@example.com",
            "subject": "Test Email",
            "key_points": ["Test point 1", "Test point 2"]
        }
    
    @staticmethod
    def generate_draft_reply(greeting="Dear Test User", content="Thank you for your email."):
        """Generate draft reply"""
        return f"""{greeting},

{content}

Best regards,
User"""


@pytest.fixture
def test_data_generator():
    """Provide test data generator"""
    return TestDataGenerator()


# Performance monitoring
@pytest.fixture
def performance_monitor():
    """Monitor test performance"""
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        def assert_duration_under(self, max_seconds):
            assert self.duration is not None, "Monitor was not started/stopped"
            assert self.duration < max_seconds, \
                f"Test took {self.duration:.2f}s, expected under {max_seconds}s"
    
    return PerformanceMonitor()