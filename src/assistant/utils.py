# %%

import datetime
import os
import pymupdf
import boto3


def process_path_or_email(path_or_text: str) -> str:
    """
    Processes a file path or raw email content.

    Args:
        path_or_email (str): Path to the file or raw email content.

    Returns:
        str: Processed text.
    """
    if os.path.isfile(path_or_text):
        print("File found, extracting text...")
        return extract_text(path_or_text)
    else:
        print("File not found, assuming input is raw email content.")
        return path_or_text


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using the PyMuPDF library.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_text(file_path: str) -> str:
    """
    Extracts text from a file based on its type.
    """
    if file_path.endswith(".pdf"):
        print("Extracting text from PDF...")
        return extract_text_from_pdf(file_path)
    else:
        print("Extracting text from non-PDF file...")
        with open(file_path, "r") as f:
            return f.read()


def make_now_filename() -> str:
    """
    Generates a filename based on the current date and time.

    Returns:
        str: Filename in the format 'draft_YYYYMMDD_HHMMSS.txt'.
    """
    now = datetime.datetime.now()
    return f"draft_{now.strftime('%Y%m%d_%H%M%S')}.txt"


def save_draft_to_file(draft: str, filepath=None) -> None:
    """
    Saves the draft text to a file.
    If a filepath is not provided, the file is named according to the current date and time,
    in a directory named 'drafts' inside the home directory.

    Args:
        draft (str): The draft text to save.
    """

    if filepath is None:
        # Ensure the drafts directory, located inside of the home directory, exists
        drafts_dir = os.path.join(os.path.expanduser("~"), "drafts")
        os.makedirs(drafts_dir, exist_ok=True)

        filename = make_now_filename()
        filepath = os.path.join(drafts_dir, filename)

    # Save the draft to the
    print(f"Saving draft to {filepath}...")
    with open(filepath, "w") as f:
        f.write(draft)


def save_draft_to_s3(draft: str, bucket_name: str, filepath=None) -> None:
    """
    Saves the draft text to an AWS S3 bucket.

    Args:
        draft (str): The draft text to save.
        bucket_name (str): The S3 bucket name.
        filepath (str, optional): The S3 object name (key) where the draft will be saved.
        If None, a filename based on the current date and time will be used.
    """
    print(f"Attempting to save draft to S3 bucket: {bucket_name}")
    
    # Convert string to bytes
    draft_bytes = draft.encode("utf-8")

    if filepath is None:
        filename = make_now_filename()
        filepath = os.path.join("drafts", filename).replace("\\", "/")  # Ensure forward slashes for S3
    
    print(f"S3 key will be: {filepath}")

    try:
        s3 = boto3.client("s3")
        print("S3 client created successfully")
        
        # Check if bucket exists and is accessible
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"Bucket {bucket_name} is accessible")
        except Exception as bucket_error:
            print(f"Warning: Cannot access bucket {bucket_name}: {bucket_error}")
        
        # Attempt to save the object
        s3.put_object(Bucket=bucket_name, Key=filepath, Body=draft_bytes)
        print(f"Draft saved successfully to s3://{bucket_name}/{filepath}")
        
    except Exception as e:
        error_msg = f"Failed to save draft to S3: {type(e).__name__}: {str(e)}"
        print(error_msg)
        
        # Provide more specific error information
        if "NoCredentialsError" in str(type(e)):
            print("AWS credentials not found. Please configure AWS credentials.")
        elif "AccessDenied" in str(e):
            print("Access denied. Check your AWS permissions for S3.")
        elif "NoSuchBucket" in str(e):
            print(f"Bucket '{bucket_name}' does not exist or is not accessible.")
        
        raise Exception(error_msg)


def extract_email_content_from_response(llm_response: str) -> str:
    """
    Extracts the actual email content from an LLM response that may contain
    explanatory text before the email.
    
    Args:
        llm_response (str): The full LLM response
        
    Returns:
        str: Just the email content without explanatory text
    """
    lines = llm_response.strip().split('\n')
    
    # Look for common patterns that indicate the start of the actual email
    email_start_patterns = [
        'Hi ',
        'Hello ',
        'Dear ',
        'Subject:',
        'To:',
        'From:',
        'Thank you',
        'Thanks',
        'I hope',
        'I am writing',
        'I would like',
        'Please',
        'Regarding',
        'Re:',
        'Best regards',
        'Sincerely',
        'Kind regards'
    ]
    
    # Find the first line that looks like it starts an email
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped:  # Skip empty lines
            # Check if this line starts with any email pattern
            for pattern in email_start_patterns:
                if line_stripped.startswith(pattern):
                    # Return everything from this line onwards
                    return '\n'.join(lines[i:]).strip()
            
            # Also check if the line contains a colon (like "Here's a draft:" or "Here's the response:")
            # and the next non-empty line might be the email start
            if ':' in line_stripped and i < len(lines) - 1:
                # Look at the next few lines to see if they start an email
                found_email_pattern = False
                for j in range(i + 1, min(i + 4, len(lines))):
                    next_line = lines[j].strip()
                    if next_line:
                        for pattern in email_start_patterns:
                            if next_line.startswith(pattern):
                                return '\n'.join(lines[j:]).strip()
                        # If we found a non-empty line but no email pattern, break
                        found_email_pattern = False
                        break
                # If we found a colon but no email pattern follows, continue to other methods
                if not found_email_pattern:
                    continue
    
    # If no clear email start pattern is found, look for the last paragraph
    # that doesn't contain explanatory words AND looks like email content
    explanatory_words = ['draft', 'response', 'reply', 'based on', 'here\'s', 'here is', 'refined', 'professional', 'think about', 'situation']
    
    # Split into paragraphs (double newlines)
    paragraphs = llm_response.strip().split('\n\n')
    
    # If we have multiple paragraphs and the first one contains explanatory language,
    # check if any subsequent paragraphs look like email content
    if len(paragraphs) > 1:
        first_paragraph_lower = paragraphs[0].lower()
        first_is_explanatory = any(word in first_paragraph_lower for word in explanatory_words)
        
        if first_is_explanatory:
            # Find the first paragraph that doesn't seem explanatory AND has email-like characteristics
            for i, paragraph in enumerate(paragraphs[1:], 1):  # Start from second paragraph
                paragraph_lower = paragraph.lower()
                is_explanatory = any(word in paragraph_lower for word in explanatory_words)
                
                # Check if paragraph has email-like characteristics
                has_email_characteristics = any(pattern.lower() in paragraph_lower for pattern in email_start_patterns)
                
                if not is_explanatory and len(paragraph.strip()) > 10 and has_email_characteristics:
                    # Return this paragraph and everything after it
                    start_index = llm_response.find(paragraph)
                    if start_index != -1:
                        return llm_response[start_index:].strip()
    
    # Fallback: return the original response if we can't identify the email content
    return llm_response.strip()


# %%
