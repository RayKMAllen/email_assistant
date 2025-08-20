"""
Main AWS Bedrock session manager.
Summarizes, extracts data from emails, drafts replies.
Maintains conversation context in session.
"""
# %%

import boto3
import json
import pprint
from botocore.exceptions import ClientError
import configparser
import os

from assistant.utils import process_path_or_email, save_draft_to_file, save_draft_to_s3
from assistant.prompts_params import (
    DRAFT_PREFIX,
    EXTRACT_PREFIX,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
)

# Load configuration from config.config in the root directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
config_path = os.path.join(ROOT_DIR, "config.config")

# Load config
config = configparser.ConfigParser()
config.read(config_path)

# Access values from [DEFAULT]
MODEL_ID = config["DEFAULT"]["model_id"]
BUCKET_NAME = config["DEFAULT"]["bucket_name"]


class EmailLLMProcessor:
    """
    Manages a session with AWS Bedrock for email processing.
    Handles summarization, data extraction, and reply drafting.
    Maintains conversation context.
    """

    def __init__(self):
        self.history = []
        self.client = boto3.client("bedrock")
        self.runtime = boto3.client("bedrock-runtime")
        self.model_id = MODEL_ID

        self.text = None  # placeholder for email text
        self.key_info = None  # placeholder for key info extraction
        self.last_draft = None  # placeholder for most recent draft reply

    def load_text(self, path_or_text):
        self.text = process_path_or_email(path_or_text)

    def send_prompt(self, prompt: str):
        """
        Sends a prompt to the Bedrock model and returns the response.

        Args:
            prompt (str): The prompt to send.

        Returns:
            str: The model's response.
        """

        # Add the prompt to the conversation history
        self.history.append({"role": "user", "content": prompt})

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "messages": [{"role": "user", "content": prompt}],
            }
        )

        accept = "application/json"
        contentType = "application/json"

        try:
            response = self.runtime.invoke_model(
                modelId=self.model_id,
                body=body,
                accept=accept,
                contentType=contentType,
            )
        except ClientError as e:
            raise Exception(f"Error invoking model: {e}")

        try:
            output = json.loads(response["body"].read().decode("utf-8"))
            output_text = output["content"][0]["text"]
        except Exception as e:
            raise Exception(f"Failed to parse model response: {e}")

        # Add the response to the conversation history
        self.history.append({"role": "assistant", "content": output_text})

        return output_text

    def extract_key_info(self):
        """
        Extracts key information from the email exchange and stores it in self.key_info.
        """

        prompt = EXTRACT_PREFIX + self.text

        key_info_string = self.send_prompt(prompt)

        # format the key info string as dict
        if "json" in key_info_string.split("\n")[0]:
            key_info_string = "\n".join(key_info_string.split("\n")[1:-1])

        try:
            key_info = json.loads(key_info_string)
            print("Key info extracted:")
            pprint.pp(key_info)
            self.key_info = key_info
        except json.JSONDecodeError:
            error_message = "Failed to parse key information from the response."
            raise Exception(error_message)

    def draft_reply(self, tone=None) -> str:
        """
        Drafts a reply to the email exchange based on the extracted text.

        Args:
            tone (str): Optional. The tone of the reply (e.g., "formal" etc.).

        Returns:
            str: The drafted reply.
        """

        tone_prompt = f" using a {tone} tone" if tone else ""
        prompt = DRAFT_PREFIX.format(tone_prompt) + self.text

        draft = self.send_prompt(prompt)

        self.last_draft = draft

        return draft

    def refine(self, instructions: str, full_history: bool = False) -> str:
        """
        Refines the last draft reply based on additional instructions.

        Args:
            instructions (str): Instructions for refining the draft.

        Returns:
            str: The refined draft reply.
        """

        if not self.last_draft:
            print("No draft reply to refine, drafting first.")
            _ = self.draft_reply()

        if full_history:
            # Include the full assistant conversation history in the prompt
            prompt = f"Refine the following draft reply based on these instructions and the subsequent history of prompts and responses: {instructions}\n\nDraft:\n{self.last_draft}\n\nAssistant conversation History:\n"
            for message in self.history:
                prompt += f"{message['role'].capitalize()}: {message['content']}\n"
        else:
            # Use the last draft and summary for refinement
            prompt = f"Refine the following draft reply based on these instructions and the subsequent summary: {instructions}\n\nDraft:\n{self.last_draft}\n\nSummary:\n{self.key_info.get('summary', '')}"

        draft = self.send_prompt(prompt)

        self.last_draft = draft

        return draft

    def save_draft(self, filepath=None, cloud=False) -> None:
        """
        Saves the last draft reply.
        If cloud is True, saves to AWS S3, otherwise saves to a local file.
        If a filepath is not provided, the file is named according to the current date and time,
        in a directory named 'drafts'.
        """

        if not self.last_draft:
            print("No draft reply to save.")
            return

        if cloud:
            save_draft_to_s3(
                self.last_draft, bucket_name=BUCKET_NAME, filepath=filepath
            )
        else:
            save_draft_to_file(self.last_draft, filepath)


# %%
