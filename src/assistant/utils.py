#%%

import datetime
import os
import pymupdf

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

def save_draft_to_file(draft: str) -> None:
    """
    Saves the draft text to a file.
    The file is named according to the current date and time, in a directory named 'drafts'.
    
    Args:
        draft (str): The draft text to save.
    """
    
    # Ensure the drafts directory exists
    os.makedirs("drafts", exist_ok=True)
    
    # Create a filename based on the current date and time
    now = datetime.datetime.now()
    filename = f"draft_{now.strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Save the draft to the file
    with open(os.path.join("drafts", filename), "w") as f:
        f.write(draft)

# %%
