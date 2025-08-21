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
    else:
        # Expand user path (e.g., ~/dds -> /home/user/dds) and ensure directory exists
        filepath = os.path.expanduser(filepath)
        directory = os.path.dirname(filepath)
        if directory:  # Only create directory if filepath contains a directory component
            os.makedirs(directory, exist_ok=True)

    # Save the draft to the file
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
        If filepath is a directory (ends with /), a timestamped filename will be added.
    """
    print(f"Attempting to save draft to S3 bucket: {bucket_name}")
    
    # Convert string to bytes
    draft_bytes = draft.encode("utf-8")

    if filepath is None:
        filename = make_now_filename()
        filepath = os.path.join("drafts", filename).replace("\\", "/")  # Ensure forward slashes for S3
    else:
        # If filepath is provided, ensure it's properly formatted for S3
        filepath = filepath.replace("\\", "/")  # Ensure forward slashes for S3
        
        # If filepath ends with / or is just a directory name, add a timestamped filename
        if filepath.endswith("/") or ("." not in os.path.basename(filepath) and "/" not in filepath.rstrip("/")):
            filename = make_now_filename()
            if not filepath.endswith("/"):
                filepath += "/"
            filepath = filepath + filename
    
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


# %%


