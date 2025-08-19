"""
Main AWS Bedrock session manager.
Summarizes, extracts data from emails, drafts replies.
Maintains conversation context in session.
"""
#%%

import boto3
import json
import pprint
from botocore.exceptions import ClientError

from assistant.utils import process_path_or_email, save_draft_to_file

model_id = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0"

SUMMARIZE_PREFIX = "Summarize the following email exchange in 2-3 sentences:\n\n"
EXTRACT_PREFIX = "Extract the key information: sender name, receiver name, sender contact details, receiver contact details,\
        subject, summary (2-3 sentences), in JSON format, from the following email exchange:\n\n"
DRAFT_PREFIX = "Draft a reply to the following email exchange{}:\n\n"

class BedrockSession:
    """
    Manages a session with AWS Bedrock for email processing.
    Handles summarization, data extraction, and reply drafting.
    Maintains conversation context.
    """
    
    def __init__(self):
        self.history = []
        self.client = boto3.client('bedrock')
        self.runtime = boto3.client('bedrock-runtime')
        self.model_id = model_id

        self.text = None  # placeholder for email text
        self.key_info = None   # placeholder for key info extraction
        self.last_draft = None # placeholder for most recent draft reply
    
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
        self.history.append(
            {"role": "user", "content": prompt}
            )

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "temperature": 0.5,
            "top_p": 0.9,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })

        accept = 'application/json'
        contentType = 'application/json'

        try:
            response = self.runtime.invoke_model(
                modelId=self.model_id,
                body=body,
                accept=accept,
                contentType=contentType,
            )
        except ClientError as e:
            raise Exception(f"Error invoking model: {e}")
        
        output = json.loads(response["body"].read().decode("utf-8"))
        output_text = output["content"][0]["text"]

        # Add the response to the conversation history
        self.history.append(
            {"role": "assistant", "content": output_text}
            )

        return output_text

    def extract_key_info(self):
        """
        Extracts key information from the email exchange and stores it in self.key_info.
        """

        prompt = EXTRACT_PREFIX + self.text

        key_info_string =  self.send_prompt(prompt)

        # format the key info string as dict
        if 'json' in key_info_string.split('\n')[0]:
            key_info_string = '\n'.join(key_info_string.split('\n')[1:-1])
        
        try:
            key_info = json.loads(key_info_string)
            print('Key info extracted:')
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

    def save_draft(self, filepath) -> None:
        """
        Saves the last draft reply to a file.
        The file is named according to the current date and time, in a directory named 'drafts'.
        """
        
        if not self.last_draft:
            print("No draft reply to save.")
            return
        
        save_draft_to_file(self.last_draft, filepath)    


# %%
