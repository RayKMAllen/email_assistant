"""
Main AWS Bedrock session manager.
Summarizes, extracts data from emails, drafts replies.
Maintains conversation context in sesion.
"""
#%%

from dotenv import load_dotenv
import boto3
import json
import os

from disk import extract_text, save_draft_to_file

load_dotenv()
AWS_BEARER_TOKEN_BEDROCK = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

aws_access_key_id = os.environ['AWS_KEY']
aws_secret_access_key = os.environ['AWS_SEC_KEY']

model_id = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0"

SUMMARIZE_PREFIX = "Summarize the following email exchange in 2-3 sentences:\n\n"
EXTRACT_PREFIX = "Extract the key information: sender name, receiver name, sender contact details, receiver contact details,\
        subject, summary (2-3 sentences), in JSON format, from the following email exchange:\n\n"
DRAFT_PREFIX = "Draft a reply to the following email exchange:\n\n"

class BedrockSession:
    """
    Manages a session with AWS Bedrock for email processing.
    Handles summarization, data extraction, and reply drafting.
    Maintains conversation context.
    """
    
    def __init__(self, text_file_path=None):
        self.token = AWS_BEARER_TOKEN_BEDROCK
        self.history = {
            "user": [],
            "assistant": []
        }
        self.client = boto3.client(
            'bedrock',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
            )
        self.runtime = boto3.client('bedrock-runtime')
        self.model_id = model_id

        if text_file_path:
            self.load_text(text_file_path)
        self.key_info = None   # placeholder for key info extraction
    
    def load_text(self, file_path):

        self.text = extract_text(file_path)
    
    def send_prompt(self, prompt: str):
        """
        Sends a prompt to the Bedrock model and returns the response.
        
        Args:
            prompt (str): The prompt to send.
        
        Returns:
            str: The model's response.
        """

        # Add the prompt to the conversation history
        self.history["user"].append(prompt)

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

        response = self.runtime.invoke_model(
            modelId=self.model_id,
            body=body,
            accept=accept,
            contentType=contentType,
        )
        
        output = json.loads(response["body"].read().decode("utf-8"))
        output_text = output["content"][0]["text"]

        # Add the response to the conversation history
        self.history["assistant"].append(output_text)

        return output_text

    def summarize(self, text: str) -> str:

        prompt = SUMMARIZE_PREFIX + text

        return self.send_prompt(prompt)

    def extract_key_info(self, text: str) -> str:

        prompt = EXTRACT_PREFIX + text

        key_info_string =  self.send_prompt(prompt)

        # format the key info string as dict
        if 'json' in key_info_string.split('\n')[0]:
            key_info_string = '\n'.join(key_info_string.split('\n')[1:-1])
        
        try:
            key_info = json.loads(key_info_string)
            self.key_info = key_info
        except json.JSONDecodeError:
            error_message = {"error": "Failed to parse key information from the response."}
            print(error_message)
        
        return key_info

    def draft_reply(self, text: str) -> str:

        prompt = DRAFT_PREFIX + text

        return self.send_prompt(prompt)

    


# %%
