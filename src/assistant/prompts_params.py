"""
Model configuration parameters, prompt templates and S3 bucket name.
"""

MODEL_ID = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0"

SUMMARIZE_PREFIX = "Summarize the following email exchange in 2-3 sentences:\n\n"
EXTRACT_PREFIX = "Extract the key information: sender name, receiver name, sender contact details, receiver contact details,\
        subject, summary (2-3 sentences), in JSON format, from the following email exchange:\n\n"
DRAFT_PREFIX = "Draft a reply to the following email exchange{}:\n\n"

MAX_TOKENS = 256
TEMPERATURE = 0.3
TOP_P = 0.2

BUCKET_NAME = "raykyrleallenbucket"
