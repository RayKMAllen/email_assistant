#!/usr/bin/env python3
"""
Test script to verify that the email content extraction function works correctly.
"""

import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from assistant.utils import extract_email_content_from_response

def test_email_extraction():
    """Test the email content extraction function with various scenarios."""
    
    # Test case 1: Response with explanatory text before email
    test_case_1 = """Based on your instructions to revert to the last non-crazy draft and the summary provided, here's a refined professional response:

Hi Colin,

Thank you for your email regarding the project update. I wanted to follow up on the discussion we had last week about the timeline.

I appreciate your patience as we work through these details. Please let me know if you have any questions.

Best regards,
John"""
    
    expected_1 = """Hi Colin,

Thank you for your email regarding the project update. I wanted to follow up on the discussion we had last week about the timeline.

I appreciate your patience as we work through these details. Please let me know if you have any questions.

Best regards,
John"""
    
    # Test case 2: Response with "Here's a draft:" prefix
    test_case_2 = """I've created a professional draft response for you:

Dear Sarah,

I hope this email finds you well. I wanted to reach out regarding the upcoming meeting scheduled for next Tuesday.

Could we possibly reschedule to Wednesday afternoon? I have a conflict that just came up.

Thank you for your understanding.

Sincerely,
Mike"""
    
    expected_2 = """Dear Sarah,

I hope this email finds you well. I wanted to reach out regarding the upcoming meeting scheduled for next Tuesday.

Could we possibly reschedule to Wednesday afternoon? I have a conflict that just came up.

Thank you for your understanding.

Sincerely,
Mike"""
    
    # Test case 3: Simple email without explanatory text
    test_case_3 = """Hello Team,

Please find attached the quarterly report. Let me know if you have any questions.

Thanks,
Alex"""
    
    expected_3 = test_case_3
    
    # Run tests
    print("Testing email content extraction...")
    
    result_1 = extract_email_content_from_response(test_case_1)
    print("\nTest 1 - Expected to extract clean email:")
    print(f"Result: {repr(result_1)}")
    print(f"Expected: {repr(expected_1)}")
    print("✅ PASS" if result_1.strip() == expected_1.strip() else "❌ FAIL")
    
    result_2 = extract_email_content_from_response(test_case_2)
    print("\nTest 2 - Expected to extract clean email:")
    print(f"Result: {repr(result_2)}")
    print(f"Expected: {repr(expected_2)}")
    print("✅ PASS" if result_2.strip() == expected_2.strip() else "❌ FAIL")
    
    result_3 = extract_email_content_from_response(test_case_3)
    print("\nTest 3 - Expected to return original (no explanatory text):")
    print(f"Result: {repr(result_3)}")
    print(f"Expected: {repr(expected_3)}")
    print("✅ PASS" if result_3.strip() == expected_3.strip() else "❌ FAIL")
    
    print("\n" + "="*50)
    print("Email extraction function test completed!")

if __name__ == "__main__":
    test_email_extraction()