"""
Hybrid intent classification system for the conversational email agent.
Uses rule-based patterns for clear cases and LLM classification for ambiguous inputs.
"""

import re
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

from assistant.conversation_state import ConversationContext, ConversationState


@dataclass
class IntentResult:
    """Result of intent classification"""
    intent: str
    confidence: float
    parameters: Dict[str, Any]
    reasoning: str
    method: str  # 'rule_based' or 'llm_based'


class HybridIntentClassifier:
    """
    Hybrid intent classifier that uses rule-based patterns for clear cases
    and LLM classification for ambiguous inputs
    """
    
    def __init__(self, email_processor=None):
        self.email_processor = email_processor
        self._setup_rule_patterns()
        self._setup_context_patterns()
    
    def _setup_rule_patterns(self):
        """Define rule-based patterns for common intents"""
        self.intent_patterns = {
            'LOAD_EMAIL': {
                'patterns': [
                    r'here.s an email',
                    r'here is an email',
                    r'process this email',
                    r'i have an email',
                    r'can you help with this email',
                    r'process.*email',    # General process email patterns
                    r'load.*email',       # Load email patterns
                    r'analyze.*email',    # Analyze email patterns
                    r'from:.*to:.*subject:',  # email format indicators
                    r'subject:.*from:',  # alternative email format
                    r'from:.*\n.*to:.*\n.*subject:',  # Multi-line email headers
                    r'from:.*\n.*subject:.*\n.*to:',  # Alternative order
                    r'to:.*\n.*from:.*\n.*subject:',  # Another order
                    r'dear.*sincerely|regards|best',  # email content patterns
                    r'^from:\s*\S+@\S+',  # Email starting with From: header
                    r'^to:\s*\S+@\S+',    # Email starting with To: header
                    r'^subject:',         # Email starting with Subject: header
                    r'process.*file',     # File processing requests
                    r'load.*file',        # Load file requests
                    r'analyze.*file',     # Analyze file requests
                    r'help with.*file',   # Help with file requests
                    r'here.s.*file',      # Here's a file
                    r'(?:process|load|analyze|open|read).*\.(pdf|txt|doc|docx|eml)',  # Load specific file types with action verbs
                ],
                'confidence': 0.9
            },
            'DRAFT_REPLY': {
                'patterns': [
                    r'draft.*reply',
                    r'write.*response',
                    r'help.*respond',
                    r'need to reply',
                    r'create.*reply',
                    r'compose.*response',
                    r'draft.*email',
                    r'try.*draft',
                    r'draft.*again',
                    r'try.*drafting',
                    r'draft.*retry',
                    r'retry.*draft',
                ],
                'confidence': 0.85
            },
            'REFINE_DRAFT': {
                'patterns': [
                    r'make it more (formal|casual|professional|friendly|polite|concise)',
                    r'make it (formal|casual|professional|friendly|polite|concise)',  # Without "more"
                    r'change.*tone',
                    r'revise.*draft',
                    r'improve.*reply',
                    r'make it (shorter|longer|more concise)',
                    r'add.*meeting',
                    r'include.*availability',
                    r'more (professional|formal)',
                    r'be more (polite|formal|casual|professional)',
                ],
                'confidence': 0.8
            },
            'SAVE_DRAFT': {
                'patterns': [
                    r'^save$',  # Simple "save" command
                    r'save.*draft',
                    r'save.*reply',
                    r'export.*file',
                    r'keep.*draft',
                    r'save it',
                    r'save this',
                    r'save.*cloud',
                    r'save.*s3',
                    r'save.*aws',
                    r'save.*locally',
                    r'save to cloud',
                    r'save to s3',
                    r'save to aws',
                    r'save locally',
                    r'save in.*cloud',
                    r'cloud.*storage',
                    r'upload.*draft',
                    r'upload.*cloud',
                    r'save\s+to\s+.*\.(txt|doc|docx|pdf|eml)',  # Save to file with extension
                    r'save\s+as\s+.*\.(txt|doc|docx|pdf|eml)',  # Save as file with extension
                    r'filepath?\s*:\s*.*\.(txt|doc|docx|pdf|eml)',  # filepath: /path/file.ext
                    r'path\s*:\s*.*\.(txt|doc|docx|pdf|eml)',  # path: /path/file.ext
                    r'save.*(?:the\s+)?(?:draft|reply|response|email).*to.*\.(txt|doc|docx|pdf|eml)',  # Save with file extension - more specific
                ],
                'confidence': 0.95  # Increased confidence to beat LOAD_EMAIL
            },
            'EXTRACT_INFO': {
                'patterns': [
                    r'what are.*key details',
                    r'show.*summary',
                    r'extract.*information',
                    r'who sent.*email',
                    r'what.s.*about',
                    r'key information',
                    r'^summary$',
                    r'show.*info',
                    r'key.*details',
                    r'what.*summary',
                ],
                'confidence': 0.8
            },
            'GENERAL_HELP': {
                'patterns': [
                    r'^help$',
                    r'^what can you do',
                    r'how does this work',
                    r'what are your capabilities',
                    r'how do i',
                    r'explain',
                ],
                'confidence': 0.9
            },
            'CONTINUE_WORKFLOW': {
                'patterns': [
                    r'^yes[!.]*$',
                    r'^ok[!.]*$',
                    r'^okay[!.]*$',
                    r'^continue[!.]*$',
                    r'^proceed[!.]*$',
                    r'^next[!.]*$',
                    r'^go ahead[!.]*$',
                    r'sounds good',
                    r'that works',
                    r'please do',
                    r'go for it',
                    r'^sure[!.]*$',
                    r'^do it[!.]*$',
                ],
                'confidence': 0.7  # Lower confidence as context-dependent
            },
            'DECLINE_OFFER': {
                'patterns': [
                    r'^no[!.]*$',
                    r'^nope[!.]*$',
                    r'^not now[!.]*$',
                    r'^not yet[!.]*$',
                    r'^skip[!.]*$',
                    r'^skip that[!.]*$',
                    r'^skip it[!.]*$',
                    r'^no thanks[!.]*$',
                    r'^no thank you[!.]*$',
                    r'not right now',
                    r'maybe later',
                    r'not interested',
                    r'^pass[!.]*$',
                ],
                'confidence': 0.7  # Lower confidence as context-dependent
            }
        }
    
    def _setup_context_patterns(self):
        """Define context-aware pattern adjustments"""
        self.context_adjustments = {
            ConversationState.EMAIL_LOADED: {
                'yes_patterns': ['CONTINUE_WORKFLOW', 'EXTRACT_INFO'],
                'no_patterns': ['DECLINE_OFFER'],
                'default_boost': {'DRAFT_REPLY': 0.1, 'EXTRACT_INFO': 0.1}
            },
            ConversationState.INFO_EXTRACTED: {
                'yes_patterns': ['CONTINUE_WORKFLOW', 'DRAFT_REPLY'],
                'no_patterns': ['DECLINE_OFFER'],
                'default_boost': {'DRAFT_REPLY': 0.15}
            },
            ConversationState.DRAFT_CREATED: {
                'yes_patterns': ['CONTINUE_WORKFLOW', 'SAVE_DRAFT'],
                'no_patterns': ['DECLINE_OFFER'],
                'default_boost': {'SAVE_DRAFT': 0.1, 'REFINE_DRAFT': 0.1}
            },
            ConversationState.DRAFT_REFINED: {
                'yes_patterns': ['CONTINUE_WORKFLOW', 'SAVE_DRAFT'],
                'no_patterns': ['DECLINE_OFFER'],
                'default_boost': {'SAVE_DRAFT': 0.15}
            }
        }
    
    def classify(self, user_input: str, context: ConversationContext) -> IntentResult:
        """
        Classify user intent using hybrid approach
        
        Args:
            user_input: The user's message
            context: Current conversation context
            
        Returns:
            IntentResult with classified intent and metadata
        """
        # First, try rule-based classification
        rule_result = self._classify_with_rules(user_input, context)
        
        # If rule-based classification is confident, use it
        if rule_result.confidence >= 0.8:
            return rule_result
        
        # For ambiguous cases, use LLM classification if available
        if self.email_processor and rule_result.confidence < 0.6:
            llm_result = self._classify_with_llm(user_input, context)
            if llm_result.confidence > rule_result.confidence:
                return llm_result
        
        # Fall back to rule-based result or clarification needed
        if rule_result.confidence > 0.3:
            return rule_result
        else:
            # If we have an LLM processor available, try LLM classification as final fallback
            if self.email_processor:
                llm_fallback = self._classify_with_llm(user_input, context)
                if llm_fallback.confidence > 0.4:  # Lower threshold for fallback
                    llm_fallback.reasoning = f"Fallback LLM classification: {llm_fallback.reasoning}"
                    return llm_fallback
            
            return IntentResult(
                intent='CLARIFICATION_NEEDED',
                confidence=0.9,
                parameters={
                    'original_input': user_input,
                    'fallback_attempted': self.email_processor is not None
                },
                reasoning='Input is ambiguous and needs clarification',
                method='fallback'
            )
    
    def _classify_with_rules(self, user_input: str, context: ConversationContext) -> IntentResult:
        """Classify intent using rule-based patterns"""
        user_input_lower = user_input.lower().strip()
        best_match = None
        best_confidence = 0.0
        best_parameters = {}
        
        # Check for email content in input
        email_content = self._extract_email_content(user_input)
        if email_content:
            best_parameters['email_content'] = email_content
        
        # Check each intent pattern
        for intent, config in self.intent_patterns.items():
            confidence = 0.0
            matched_patterns = []
            
            for pattern in config['patterns']:
                if re.search(pattern, user_input_lower):
                    confidence = max(confidence, config['confidence'])
                    matched_patterns.append(pattern)
            
            # Apply context-based adjustments
            adjusted_confidence = self._apply_context_adjustments(
                intent, confidence, user_input_lower, context
            )
            confidence = max(confidence, adjusted_confidence)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = intent
                best_parameters.update({
                    'matched_patterns': matched_patterns,
                    'tone': self._extract_tone(user_input_lower),
                    'refinement_instructions': self._extract_refinement_instructions(user_input),  # Use original case
                    'cloud': self._extract_cloud_preference(user_input_lower),
                    'filepath': self._extract_filepath(user_input_lower)
                })
        
        return IntentResult(
            intent=best_match or 'CLARIFICATION_NEEDED',
            confidence=best_confidence,
            parameters=best_parameters,
            reasoning=f"Rule-based match with confidence {best_confidence}",
            method='rule_based'
        )
    
    def _apply_context_adjustments(self, intent: str, confidence: float,
                                 user_input: str, context: ConversationContext) -> float:
        """Apply context-based confidence adjustments"""
        current_state = context.current_state
        
        if current_state not in self.context_adjustments:
            return confidence
        
        adjustments = self.context_adjustments[current_state]
        user_input_clean = user_input.strip().lower()
        
        # Handle simple affirmative responses in context
        if user_input_clean in ['yes', 'ok', 'okay', 'continue', 'proceed', 'sure', 'please do', 'go for it', 'do it']:
            if intent == 'CONTINUE_WORKFLOW':
                return 0.95  # High confidence for yes responses to offers
        
        # Handle simple negative responses in context
        if user_input_clean in ['no', 'nope', 'not now', 'not yet', 'skip', 'skip that', 'skip it', 'no thanks', 'no thank you', 'pass']:
            if intent == 'DECLINE_OFFER':
                return 0.95  # High confidence for no responses to offers
        
        # Apply default boosts for likely intents in current state
        if 'default_boost' in adjustments and intent in adjustments['default_boost']:
            confidence += adjustments['default_boost'][intent]
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _extract_email_content(self, user_input: str) -> Optional[str]:
        """Extract email content or file path from user input if present"""
        # First, look for file paths in natural language
        file_path = self._extract_file_path(user_input)
        if file_path:
            return file_path
        
        # Look for email-like patterns
        email_indicators = [
            r'from:.*to:.*subject:',
            r'subject:.*from:',
            r'from:.*\n.*to:.*\n.*subject:',  # Multi-line email headers
            r'from:.*\n.*subject:.*\n.*to:',  # Alternative order
            r'to:.*\n.*from:.*\n.*subject:',  # Another order
            r'dear.*sincerely|regards|best',
        ]
        
        for pattern in email_indicators:
            if re.search(pattern, user_input, re.IGNORECASE | re.DOTALL):
                # If it looks like email content, return the whole input
                return user_input.strip()
        
        return None
    
    def _extract_file_path(self, user_input: str) -> Optional[str]:
        """Extract file path from natural language input"""
        # Patterns to match file paths in natural language
        file_path_patterns = [
            r'(?:process|load|open|read|analyze|help with|work with)\s+(?:this\s+)?(?:file|email|document):\s*([^\s]+)',
            r'(?:here.s|here is)\s+(?:a\s+)?(?:file|email|document):\s*([^\s]+)',
            r'(?:file|email|document)\s+(?:is|at|located at):\s*([^\s]+)',
            r'(?:load|process|analyze)\s+([^\s]+\.(?:docx|pdf|txt|eml|doc))',  # Longer extensions first
            r'([^\s]+\.(?:docx|pdf|txt|eml|doc))(?:\s|$)',  # Just a file with extension, longer first
            r'(?:help with|work with|process|load|analyze)\s+[\'"]([^\'\"]+)[\'"]',  # Quoted filenames
        ]
        
        for pattern in file_path_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                file_path = match.group(1).strip()
                # Remove quotes if present
                file_path = file_path.strip('"\'')
                return file_path
        
        return None
    
    def _extract_tone(self, user_input: str) -> Optional[str]:
        """Extract requested tone from user input"""
        tone_patterns = {
            'formal': r'formal|professional',
            'casual': r'casual|informal|friendly',
            'concise': r'concise|brief|short',
            'polite': r'polite|courteous',
        }
        
        for tone, pattern in tone_patterns.items():
            if re.search(pattern, user_input):
                return tone
        
        return None
    
    def _extract_refinement_instructions(self, user_input: str) -> Optional[str]:
        """Extract specific refinement instructions"""
        refinement_patterns = [
            r'make it (?:more|less) \w+',
            r'add \w+',
            r'include \w+',
            r'change \w+',
            r'remove \w+',
        ]
        
        instructions = []
        for pattern in refinement_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            instructions.extend(matches)
        
        return ' '.join(instructions) if instructions else user_input
    
    def _extract_cloud_preference(self, user_input: str) -> bool:
        """Extract whether user wants to save to cloud/S3"""
        cloud_patterns = [
            r'save.*cloud',
            r'save.*s3',
            r'cloud.*storage',
            r'upload.*draft',
            r'save.*aws',
            r'to.*cloud',
            r'in.*cloud'
        ]
        
        for pattern in cloud_patterns:
            if re.search(pattern, user_input):
                return True
        
        return False
    
    def _extract_filepath(self, user_input: str) -> Optional[str]:
        """Extract specific filepath if mentioned"""
        # Look for filepath patterns like "save to /path/file.txt" or "save as filename.txt"
        # But exclude cloud-related terms
        cloud_terms = ['cloud', 's3', 'aws', 'bucket']
        
        filepath_patterns = [
            r'save\s+to\s+([^\s]+\.(?:txt|doc|docx|pdf|eml))',  # "save to file.ext" - must have extension
            r'save\s+as\s+([^\s]+)',      # "save as filename.txt"
            r'save\s+(?:to|as)\s+([^\s]+\.txt)',
            r'save\s+(?:to|as)\s+([^\s]+\.pdf)',
            r'filepath?\s*:\s*([^\s]+)',
            r'path\s*:\s*([^\s]+)',
            r'save\s+to\s+([/\\][\w/\\.-]+)',  # Absolute paths starting with / or \
            r'save\s+to\s+([\w.-]+[/\\][\w/\\.-]+)',  # Relative paths with directory separators
            r'(?:save.*(?:cloud|s3|aws).*)?in\s+dir(?:ectory)?\s+([^\s]+)',  # "in dir [directory]" or "in directory [directory]"
            r'(?:save.*(?:cloud|s3|aws).*)?to\s+dir(?:ectory)?\s+([^\s]+)',  # "to dir [directory]" or "to directory [directory]"
            r'save.*in\s+dir(?:ectory)?\s+([^\s]+)',  # "save in dir [directory]" - more general
            r'save.*to\s+dir(?:ectory)?\s+([^\s]+)',  # "save to dir [directory]" - more general
        ]
        
        for pattern in filepath_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                filepath = match.group(1).strip()
                # Remove quotes if present
                filepath = filepath.strip('"\'')
                
                # Skip if it's a cloud-related term
                if filepath.lower() in cloud_terms:
                    continue
                
                # For directory patterns, ensure we return a directory path format
                if 'dir' in pattern:
                    # If it's just a directory name, format it as a directory path
                    if not filepath.endswith('/') and '/' not in filepath and '\\' not in filepath:
                        filepath = f"{filepath}/"
                    
                return filepath
        
        return None
    
    def _classify_with_llm(self, user_input: str, context: ConversationContext) -> IntentResult:
        """Classify intent using LLM when rule-based classification is uncertain"""
        if not self.email_processor:
            return IntentResult(
                intent='CLARIFICATION_NEEDED',
                confidence=0.5,
                parameters={},
                reasoning='LLM classification not available',
                method='fallback'
            )
        
        # Check if email_processor has send_prompt method (avoid Mock issues in tests)
        if not hasattr(self.email_processor, 'send_prompt') or not callable(getattr(self.email_processor, 'send_prompt')):
            return IntentResult(
                intent='CLARIFICATION_NEEDED',
                confidence=0.5,
                parameters={},
                reasoning='LLM classification method not available',
                method='fallback'
            )
        
        # Create classification prompt
        prompt = self._create_classification_prompt(user_input, context)
        
        try:
            response = self.email_processor.send_prompt(prompt)
            
            # Check if response is a string (avoid Mock object issues in tests)
            if not isinstance(response, str):
                return IntentResult(
                    intent='CLARIFICATION_NEEDED',
                    confidence=0.5,
                    parameters={},
                    reasoning='LLM classification returned invalid response type',
                    method='fallback'
                )
            
            result = self._parse_llm_response(response)
            # Only override method if it's not already an error_fallback
            if result.method != 'error_fallback':
                result.method = 'llm_based'
            return result
        except Exception as e:
            print(f"LLM classification failed: {e}")
            return IntentResult(
                intent='CLARIFICATION_NEEDED',
                confidence=0.5,
                parameters={'error': str(e)},
                reasoning='LLM classification failed',
                method='error_fallback'
            )
    
    def _create_classification_prompt(self, user_input: str, context: ConversationContext) -> str:
        """Create prompt for LLM intent classification"""
        valid_intents = [
            'LOAD_EMAIL', 'DRAFT_REPLY', 'EXTRACT_INFO', 'REFINE_DRAFT',
            'SAVE_DRAFT', 'GENERAL_HELP', 'CONTINUE_WORKFLOW', 'DECLINE_OFFER', 'CLARIFICATION_NEEDED'
        ]
        
        prompt = f"""
Analyze the user's message and classify their intent for an email assistant conversation.

Current conversation state: {context.current_state.value}
Recent conversation: {context.get_recent_history(3)}
User message: "{user_input}"

Classify the intent as one of: {', '.join(valid_intents)}

IMPORTANT CONTEXT RULES:
- If the user previously declined an offer (said "no") and now says something like "ok fine", "yes", "okay", this usually means CONTINUE_WORKFLOW (they changed their mind and want to proceed)
- If the current state is "info_extracted" and user says affirmative words after declining, they likely want to CONTINUE_WORKFLOW (draft a reply)
- Only use CLARIFICATION_NEEDED if the user's message is truly ambiguous and doesn't fit any workflow pattern
- Consider the natural flow: after declining a draft offer, saying "ok fine" typically means accepting the original offer

Return your response in this exact JSON format:
{{
  "intent": "INTENT_NAME",
  "confidence": 0.95,
  "parameters": {{
    "email_content": "extracted email if present",
    "tone": "formal/casual/etc if specified",
    "refinement_instructions": "specific changes requested",
    "cloud": true/false,
    "filepath": "specific filepath if mentioned"
  }},
  "reasoning": "Why this intent was chosen"
}}
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> IntentResult:
        """Parse LLM response into IntentResult"""
        try:
            # Clean up response if it has markdown formatting
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1]
            
            data = json.loads(response.strip())
            
            return IntentResult(
                intent=data.get('intent', 'CLARIFICATION_NEEDED'),
                confidence=float(data.get('confidence', 0.5)),
                parameters=data.get('parameters', {}),
                reasoning=data.get('reasoning', 'LLM classification'),
                method='llm_based'
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to parse LLM response: {e}")
            return IntentResult(
                intent='CLARIFICATION_NEEDED',
                confidence=0.3,
                parameters={'parse_error': str(e), 'raw_response': response},
                reasoning='Failed to parse LLM response',
                method='error_fallback'
            )