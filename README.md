# Email Assistant

**Email Assistant** is an interactive command-line tool for summarizing,
extracting key information, and drafting replies to emails using AWS
Bedrock LLMs. It supports loading emails from text or PDF files,
extracting structured information, and generating draft replies with
optional tone and refinement.

## Features

-   Load email content from file path (including PDF) or raw text
-   Extract key information (sender, receiver, summary, etc.) in JSON
    format
-   Draft replies with optional tone (e.g., formal, informal)
-   Refine drafted replies with additional instructions
-   Save drafts to a file (default location or custom path)
-   Interactive CLI

## Requirements

-   Python 3.8+
-   AWS credentials with access to Bedrock
-   [click](https://palletsprojects.com/p/click/)
-   [python-dotenv](https://pypi.org/project/python-dotenv/)
-   [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) (for PDF
    extraction)
-   [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

## Installation

1.  Clone the repository:
    git clone https://github.com/RayKMAllen/email_assistant.git
    
2.  Set up your `.env` file in the project root:
    `AWS_KEY=your_aws_access_key     AWS_SEC_KEY=your_aws_secret_key`

3.  Install dependencies and the package itself: 
    
    ``` sh
    cd email_assistant
    pip install -r requirements.txt
    pip install -e .
    ```

## Usage

### CLI

Run the CLI using the installed script:

``` sh
eassistant
```


### Commands

-   `load <path_or_text>`: Load an email conversation from a file path or raw text.
-   `draft [tone]`: Draft a reply to the loaded email conversation, optionally
    specifying a tone.
-   `refine <instructions>`: Refine the drafted reply with additional
    instructions.
-   `save [filepath]`: Save the drafted reply to a file (default:
    `~/drafts/draft_<timestamp>.txt`).
-   `info`: Show extracted key information from the loaded email conversation.
-   `summary`: Show the summary extracted from the loaded email conversation.
-   `exit`: Exit the CLI.

### Example

``` sh
eassistant
Welcome! To start, please load the email path or content.

> load example.pdf
File found, extracting text...
Extracting text from PDF...
Email content loaded successfully.
Key info extracted:
{'sender_name': 'Joe Jones',
 'receiver_name': 'Sam Smith',
 'sender_contact_details': {'email': 'joe@joe.com',
                            'web': 'www.joe.com'},
 'receiver_contact_details': {'email': 'sammy@gmail.com'},
 'subject': 'Catching Up',
 'summary': 'Joe Jones from Joe Inc. reached out to Sam Smith to discuss a merger with Sammy Ltd. Sam responded saying he would be keen to discuss and suggested catching up on Friday.'}

Type 'draft' to draft a reply, with an optional tone argument (e.g. 'draft formal').

> draft quick
Hi Sam,

Friday works for me. You can reach me at +1 234 56789876.

Thanks,
Joe

Type 'save' to save the draft to a file.
Type 'refine' with additional instructions to refine the draft.

> refine make it shorter and more polite
Hi Sam,

Friday would be perfect! Feel free to reach me at +1 234 56789876 whenever it's convenient for you. I'm looking forward to chatting about the opportunity!

Thanks so much,
Joe

Type 'save' to save the refined draft to a file.
Type 'refine' with additional instructions to further refine.

> save
Saving draft to C:\Users\joe\drafts\draft_20250818_170913.txt...
Draft saved to default location.

> exit
👋 Goodbye!
```

## Testing

Run tests with:

``` sh
pytest
```

## Project Structure

-   [`src/assistant/llm_session.py`](src/assistant/llm_session.py):
    Bedrock session and core logic
-   [`src/assistant/utils.py`](src/assistant/utils.py): Utilities for
    file and PDF handling
-   [`src/cli/cli.py`](src/cli/cli.py): Command-line interface
-   [`tests/`](tests/): Unit tests



------------------------------------------------------------------------

**Author:** Raymond Allen
