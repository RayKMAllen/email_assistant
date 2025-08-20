"""
Comprehensive unit tests for the hybrid intent classification system.
"""

import pytest
from unittest.mock import Mock
import json

from src.assistant.intent_classifier import (
    HybridIntentClassifier,
    IntentResult
)
from src.assistant.conversation_state import (
    ConversationContext,
    ConversationState
)


class TestIntentResult:
    """Test the IntentResult dataclass"""
    
    def test_intent_result_creation(self):
        """Test creating an IntentResult"""
        result = IntentResult(
            intent="LOAD_EMAIL",
            confidence=0.9,
            parameters={"email_content": "test"},
            reasoning="Pattern match",
            method="rule_based"
        )
        
        assert result.intent == "LOAD_EMAIL"
        assert result.confidence == 0.9
        assert result.parameters == {"email_content": "test"}
        assert result.reasoning == "Pattern match"
        assert result.method == "rule_based"


class TestHybridIntentClassifier:
    """Test the HybridIntentClassifier class"""
    
    @pytest.fixture
    def classifier(self):
        """Create a classifier instance for testing"""
        return HybridIntentClassifier()
    
    @pytest.fixture
    def context(self):
        """Create a conversation context for testing"""
        return ConversationContext()
    
    @pytest.fixture
    def mock_email_processor(self):
        """Create a mock email processor"""
        mock = Mock()
        mock.send_prompt = Mock()
        return mock
    
    def test_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier.email_processor is None
        assert hasattr(classifier, 'intent_patterns')
        assert hasattr(classifier, 'context_adjustments')
        assert 'LOAD_EMAIL' in classifier.intent_patterns
        assert 'DRAFT_REPLY' in classifier.intent_patterns
    
    def test_initialization_with_email_processor(self, mock_email_processor):
        """Test classifier initialization with email processor"""
        classifier = HybridIntentClassifier(email_processor=mock_email_processor)
        assert classifier.email_processor == mock_email_processor
    
    # Test rule-based classification
    
    def test_load_email_intent_detection(self, classifier, context):
        """Test detecting LOAD_EMAIL intent"""
        test_cases = [
            "Here's an email I need help with",
            "Here is an email from John",
            "Process this email please",
            "I have an email to analyze",
            "Can you help with this email?",
            "From: john@example.com\nTo: me@example.com\nSubject: Test",
            "Dear John,\n\nThank you for your email.\n\nBest regards,\nMe"
        ]
        
        for test_input in test_cases:
            result = classifier.classify(test_input, context)
            assert result.intent == 'LOAD_EMAIL', f"Failed for input: {test_input}"
            assert result.confidence >= 0.8
            assert result.method == 'rule_based'
    
    def test_draft_reply_intent_detection(self, classifier, context):
        """Test detecting DRAFT_REPLY intent"""
        test_cases = [
            "Draft a reply",
            "Write a response",
            "Help me respond",
            "I need to reply",
            "Create a reply",
            "Compose a response",
            "Draft an email"
        ]
        
        for test_input in test_cases:
            result = classifier.classify(test_input, context)
            assert result.intent == 'DRAFT_REPLY', f"Failed for input: {test_input}"
            assert result.confidence >= 0.8
            assert result.method == 'rule_based'
    
    def test_refine_draft_intent_detection(self, classifier, context):
        """Test detecting REFINE_DRAFT intent"""
        test_cases = [
            "Make it more formal",
            "Make it more professional",
            "Change the tone to casual",
            "Revise the draft",
            "Improve the reply",
            "Make it shorter",
            "Make it more concise",
            "Add a meeting request"
        ]
        
        for test_input in test_cases:
            result = classifier.classify(test_input, context)
            assert result.intent == 'REFINE_DRAFT', f"Failed for input: {test_input}"
            assert result.confidence >= 0.7
            assert result.method == 'rule_based'
    
    def test_save_draft_intent_detection(self, classifier, context):
        """Test detecting SAVE_DRAFT intent"""
        test_cases = [
            "Save the draft",
            "Save this reply",
            "Export to file",
            "Keep the draft",
            "Save it",
            "Save this",
            "Save to cloud",
            "Save to S3",
            "Upload the draft"
        ]
        
        for test_input in test_cases:
            result = classifier.classify(test_input, context)
            assert result.intent == 'SAVE_DRAFT', f"Failed for input: {test_input}"
            assert result.confidence >= 0.8
            assert result.method == 'rule_based'
    
    def test_general_help_intent_detection(self, classifier, context):
        """Test detecting GENERAL_HELP intent"""
        test_cases = [
            "help",
            "what can you do",
            "how does this work",
            "what are your capabilities",
            "how do I use this",
            "explain your features"
        ]
        
        for test_input in test_cases:
            result = classifier.classify(test_input, context)
            assert result.intent == 'GENERAL_HELP', f"Failed for input: {test_input}"
            assert result.confidence >= 0.8
            assert result.method == 'rule_based'
    
    def test_continue_workflow_intent_detection(self, classifier, context):
        """Test detecting CONTINUE_WORKFLOW intent"""
        test_cases = [
            "yes",
            "ok",
            "okay",
            "continue",
            "proceed",
            "next",
            "go ahead",
            "sounds good",
            "that works"
        ]
        
        for test_input in test_cases:
            result = classifier.classify(test_input, context)
            assert result.intent == 'CONTINUE_WORKFLOW', f"Failed for input: {test_input}"
            assert result.confidence >= 0.7
            assert result.method == 'rule_based'
    
    # Test parameter extraction
    
    def test_tone_extraction(self, classifier, context):
        """Test extracting tone from user input"""
        test_cases = [
            ("Make it more formal", "formal"),
            ("Change to casual tone", "casual"),
            ("Make it professional", "formal"),
            ("Make it friendly", "casual"),
            ("Make it concise", "concise"),
            ("Be more polite", "polite")
        ]
        
        for test_input, expected_tone in test_cases:
            result = classifier.classify(test_input, context)
            assert result.parameters.get('tone') == expected_tone, f"Failed for: {test_input}"
    
    def test_cloud_preference_extraction(self, classifier, context):
        """Test extracting cloud storage preference"""
        test_cases = [
            ("Save to cloud", True),
            ("Save to S3", True),
            ("Upload to cloud storage", True),
            ("Save to AWS", True),
            ("Save in the cloud", True),
            ("Save locally", False),
            ("Save the draft", False)
        ]
        
        for test_input, expected_cloud in test_cases:
            result = classifier.classify(test_input, context)
            assert result.parameters.get('cloud') == expected_cloud, f"Failed for: {test_input}"
    
    def test_filepath_extraction(self, classifier, context):
        """Test extracting filepath from user input"""
        test_cases = [
            ("Save to /path/file.txt", "/path/file.txt"),
            ("Save as draft.txt", "draft.txt"),
            ("filepath: /home/user/draft.pdf", "/home/user/draft.pdf"),
            ("path: ./drafts/reply.txt", "./drafts/reply.txt")
        ]
        
        for test_input, expected_path in test_cases:
            result = classifier.classify(test_input, context)
            assert result.parameters.get('filepath') == expected_path, f"Failed for: {test_input}"
    
    def test_email_content_extraction(self, classifier, context):
        """Test extracting email content from user input"""
        email_content = """From: john@example.com
To: me@example.com
Subject: Test Email

Dear Me,

This is a test email.

Best regards,
John"""
        
        user_input = f"Here's an email I need help with: {email_content}"
        result = classifier.classify(user_input, context)
        
        assert result.intent == 'LOAD_EMAIL'
        assert 'email_content' in result.parameters
        # Should extract the full input since it contains email-like content
        assert email_content in result.parameters['email_content']
    
    # Test context-aware classification
    
    def test_context_aware_continue_workflow(self, classifier):
        """Test context-aware classification for continue workflow"""
        # In EMAIL_LOADED state, "yes" should be CONTINUE_WORKFLOW with high confidence
        context = ConversationContext()
        context.current_state = ConversationState.EMAIL_LOADED
        
        result = classifier.classify("yes", context)
        assert result.intent == 'CONTINUE_WORKFLOW'
        assert result.confidence >= 0.8
    
    def test_context_boosts_relevant_intents(self, classifier):
        """Test that context boosts confidence for relevant intents"""
        context = ConversationContext()
        
        # In DRAFT_CREATED state, save-related intents should get boosted
        context.current_state = ConversationState.DRAFT_CREATED
        result = classifier.classify("save it", context)
        assert result.intent == 'SAVE_DRAFT'
        assert result.confidence >= 0.9  # Should be boosted
    
    # Test LLM classification fallback
    
    def test_llm_classification_when_rule_based_uncertain(self, mock_email_processor, context):
        """Test LLM classification when rule-based is uncertain"""
        classifier = HybridIntentClassifier(email_processor=mock_email_processor)
        
        # Mock LLM response
        llm_response = {
            "intent": "DRAFT_REPLY",
            "confidence": 0.85,
            "parameters": {"tone": "formal"},
            "reasoning": "User wants to create a formal response"
        }
        mock_email_processor.send_prompt.return_value = json.dumps(llm_response)
        
        # Use ambiguous input that won't match rules strongly
        result = classifier.classify("I need something professional", context)
        
        assert result.intent == 'DRAFT_REPLY'
        assert result.confidence == 0.85
        assert result.method == 'llm_based'
        assert result.parameters['tone'] == 'formal'
    
    def test_llm_classification_with_markdown_response(self, mock_email_processor, context):
        """Test LLM classification with markdown-formatted response"""
        classifier = HybridIntentClassifier(email_processor=mock_email_processor)
        
        llm_response = """```json
{
  "intent": "SAVE_DRAFT",
  "confidence": 0.9,
  "parameters": {"cloud": true},
  "reasoning": "User wants to save to cloud"
}
```"""
        mock_email_processor.send_prompt.return_value = llm_response
        
        result = classifier.classify("store this somewhere safe", context)
        
        assert result.intent == 'SAVE_DRAFT'
        assert result.confidence == 0.9
        assert result.method == 'llm_based'
        assert result.parameters['cloud'] is True
    
    def test_llm_classification_parse_error_fallback(self, mock_email_processor, context):
        """Test fallback when LLM response can't be parsed"""
        classifier = HybridIntentClassifier(email_processor=mock_email_processor)
        
        # Invalid JSON response
        mock_email_processor.send_prompt.return_value = "This is not JSON"
        
        result = classifier.classify("ambiguous input", context)
        
        assert result.intent == 'CLARIFICATION_NEEDED'
        assert result.method == 'error_fallback'
        assert 'parse_error' in result.parameters
    
    def test_llm_classification_exception_handling(self, mock_email_processor, context):
        """Test exception handling in LLM classification"""
        classifier = HybridIntentClassifier(email_processor=mock_email_processor)
        
        # Mock LLM to raise exception
        mock_email_processor.send_prompt.side_effect = Exception("LLM error")
        
        result = classifier.classify("ambiguous input", context)
        
        assert result.intent == 'CLARIFICATION_NEEDED'
        assert result.method == 'error_fallback'
        assert 'error' in result.parameters
    
    # Test clarification needed cases
    
    def test_clarification_needed_for_ambiguous_input(self, classifier, context):
        """Test that ambiguous input triggers clarification"""
        ambiguous_inputs = [
            "something",
            "do that thing",
            "fix it",
            "make changes",
            "update"
        ]
        
        for test_input in ambiguous_inputs:
            result = classifier.classify(test_input, context)
            # Should either get low confidence or clarification needed
            assert result.confidence < 0.6 or result.intent == 'CLARIFICATION_NEEDED'
    
    def test_high_confidence_rule_based_skips_llm(self, mock_email_processor, context):
        """Test that high confidence rule-based results skip LLM"""
        classifier = HybridIntentClassifier(email_processor=mock_email_processor)
        
        # Clear input that should match rules with high confidence
        result = classifier.classify("draft a reply", context)
        
        assert result.intent == 'DRAFT_REPLY'
        assert result.method == 'rule_based'
        # LLM should not have been called
        mock_email_processor.send_prompt.assert_not_called()
    
    # Test file path extraction
    
    def test_file_path_extraction(self, classifier, context):
        """Test extracting file paths from natural language"""
        test_cases = [
            ("process this file: /path/to/email.txt", "/path/to/email.txt"),
            ("here's a file: email.pdf", "email.pdf"),
            ("load document.docx", "document.docx"),
            ("analyze /home/user/emails/important.eml", "/home/user/emails/important.eml"),
            ("help with 'quoted file.txt'", "quoted file.txt")
        ]
        
        for test_input, expected_path in test_cases:
            result = classifier.classify(test_input, context)
            assert result.intent == 'LOAD_EMAIL'
            assert result.parameters.get('email_content') == expected_path
    
    # Test edge cases
    
    def test_empty_input(self, classifier, context):
        """Test handling empty input"""
        result = classifier.classify("", context)
        assert result.intent == 'CLARIFICATION_NEEDED'
        assert result.confidence >= 0.9
    
    def test_whitespace_only_input(self, classifier, context):
        """Test handling whitespace-only input"""
        result = classifier.classify("   \n\t  ", context)
        assert result.intent == 'CLARIFICATION_NEEDED'
        assert result.confidence >= 0.9
    
    def test_very_long_input(self, classifier, context):
        """Test handling very long input"""
        long_input = "Here's an email " + "very " * 1000 + "long email content"
        result = classifier.classify(long_input, context)
        assert result.intent == 'LOAD_EMAIL'
        assert result.confidence >= 0.8
    
    def test_special_characters_input(self, classifier, context):
        """Test handling input with special characters"""
        special_input = "Draft a reply with émojis 🎉 and spëcial chars!"
        result = classifier.classify(special_input, context)
        assert result.intent == 'DRAFT_REPLY'
        assert result.confidence >= 0.8
    
    # Test pattern matching edge cases
    
    def test_case_insensitive_matching(self, classifier, context):
        """Test that pattern matching is case insensitive"""
        test_cases = [
            "DRAFT A REPLY",
            "Draft A Reply",
            "draft a reply",
            "DrAfT a RePlY"
        ]
        
        for test_input in test_cases:
            result = classifier.classify(test_input, context)
            assert result.intent == 'DRAFT_REPLY'
    
    def test_partial_word_matching(self, classifier, context):
        """Test that patterns don't match partial words incorrectly"""
        # "draft" should not match "redraft" or "draftsman"
        result = classifier.classify("I am a draftsman", context)
        assert result.intent != 'DRAFT_REPLY'
    
    @pytest.mark.parametrize("state,input_text,expected_intent", [
        (ConversationState.GREETING, "yes", "CONTINUE_WORKFLOW"),
        (ConversationState.EMAIL_LOADED, "continue", "CONTINUE_WORKFLOW"),
        (ConversationState.DRAFT_CREATED, "ok", "CONTINUE_WORKFLOW"),
        (ConversationState.ERROR_RECOVERY, "help", "GENERAL_HELP"),
    ])
    def test_context_dependent_classification(self, classifier, state, input_text, expected_intent):
        """Test context-dependent classification using parametrized tests"""
        context = ConversationContext()
        context.current_state = state
        
        result = classifier.classify(input_text, context)
        assert result.intent == expected_intent