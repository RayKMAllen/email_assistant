"""
Prompt templates and LLM parameters.
"""

EXTRACT_PREFIX = "Extract the key information: sender name, receiver name, sender contact details, receiver contact details,\
        subject, summary (2-3 sentences), in JSON format, from the following email exchange:\n\n"
DRAFT_PREFIX = "Draft a reply to the following email exchange{}:\n\n"

MAX_TOKENS = 1024
TEMPERATURE = 0.3
TOP_P = 0.2