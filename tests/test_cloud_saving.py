#!/usr/bin/env python3
"""
Test script to verify cloud saving functionality works correctly.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from assistant.conversational_agent import ConversationalEmailAgent

def test_cloud_saving():
    """Test the cloud saving functionality"""
    print("🧪 Testing Cloud Saving Functionality")
    print("=" * 50)
    
    # Create agent
    agent = ConversationalEmailAgent()
    
    # Test 1: Load an email
    print("\n1. Loading test email...")
    email_content = """
From: john.doe@example.com
To: me@mycompany.com
Subject: Test Email for Cloud Saving

Hi there,

This is a test email to verify that our cloud saving functionality works correctly.

Best regards,
John Doe
"""
    
    response = agent.process_user_input(f"Here's an email I need help with: {email_content}")
    print(f"✅ Email loaded: {response[:100]}...")
    
    # Test 2: Draft a reply
    print("\n2. Drafting a reply...")
    response = agent.process_user_input("Draft a professional reply")
    print(f"✅ Reply drafted: {response[:100]}...")
    
    # Test 3: Test local saving (should work)
    print("\n3. Testing local saving...")
    try:
        response = agent.process_user_input("save this draft")
        print(f"✅ Local saving: {response}")
    except Exception as e:
        print(f"❌ Local saving failed: {e}")
    
    # Test 4: Test cloud saving (this was the issue)
    print("\n4. Testing cloud saving...")
    try:
        response = agent.process_user_input("save this draft to cloud storage")
        print(f"✅ Cloud saving: {response}")
    except Exception as e:
        print(f"❌ Cloud saving failed: {e}")
    
    # Test 5: Test S3 specific saving
    print("\n5. Testing S3 specific saving...")
    try:
        response = agent.process_user_input("save this draft to S3")
        print(f"✅ S3 saving: {response}")
    except Exception as e:
        print(f"❌ S3 saving failed: {e}")

if __name__ == "__main__":
    test_cloud_saving()