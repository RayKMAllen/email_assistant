#!/usr/bin/env python3
"""
Test script to verify that session history management works correctly.
Tests that users can view and interact with all info, summaries, email texts, drafts etc. 
from the history of the session, even after loading new documents.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from assistant.conversational_agent import ConversationalEmailAgent

def test_session_history():
    """Test that session history is preserved and accessible"""
    print("🧪 Testing Session History Management")
    print("=" * 50)
    
    # Create agent
    agent = ConversationalEmailAgent()
    
    # Test 1: Load first email
    print("\n1. Loading first email...")
    email1_content = """
From: alice@company.com
To: me@mycompany.com
Subject: Project Alpha Update

Hi there,

I wanted to update you on the progress of Project Alpha. We've completed the initial phase and are ready to move to the next stage.

Best regards,
Alice Smith
"""
    
    response = agent.process_user_input(f"Here's an email I need help with: {email1_content}")
    print(f"✅ First email loaded: {response[:100]}...")
    
    # Draft a reply for first email
    print("\n2. Drafting reply for first email...")
    response = agent.process_user_input("Draft a professional reply")
    print(f"✅ First reply drafted: {response[:100]}...")
    
    # Test 2: Load second email (this should preserve the first email's history)
    print("\n3. Loading second email...")
    email2_content = """
From: bob@vendor.com
To: me@mycompany.com
Subject: Invoice #12345

Dear Customer,

Please find attached invoice #12345 for the services provided last month. Payment is due within 30 days.

Thank you,
Bob Johnson
"""
    
    response = agent.process_user_input(f"Here's another email: {email2_content}")
    print(f"✅ Second email loaded: {response[:100]}...")
    
    # Draft a reply for second email
    print("\n4. Drafting reply for second email...")
    response = agent.process_user_input("Draft a reply")
    print(f"✅ Second reply drafted: {response[:100]}...")
    
    # Test 3: View session history
    print("\n5. Testing session history view...")
    response = agent.process_user_input("show history")
    print(f"Session history response:\n{response}")
    
    # Check if both emails are mentioned in the history
    if "alice@company.com" in response.lower() or "project alpha" in response.lower():
        print("✅ First email found in history")
    else:
        print("❌ First email NOT found in history")
    
    if "bob@vendor.com" in response.lower() or "invoice" in response.lower():
        print("✅ Second email found in history")
    else:
        print("❌ Second email NOT found in history")
    
    # Test 4: View specific session
    print("\n6. Testing specific session view...")
    response = agent.process_user_input("show email 1")
    print(f"Specific session response:\n{response}")
    
    # Check if the first email details are shown
    if "alice" in response.lower() or "project alpha" in response.lower():
        print("✅ First email details retrieved successfully")
    else:
        print("❌ First email details NOT retrieved")
    
    # Test 5: Load third email and verify all history is still preserved
    print("\n7. Loading third email...")
    email3_content = """
From: carol@client.com
To: me@mycompany.com
Subject: Meeting Request

Hello,

Could we schedule a meeting next week to discuss the new contract terms?

Best,
Carol Davis
"""
    
    response = agent.process_user_input(f"Process this email: {email3_content}")
    print(f"✅ Third email loaded: {response[:100]}...")
    
    # Test 6: Verify all three emails are in history
    print("\n8. Verifying complete session history...")
    response = agent.process_user_input("list all emails")
    print(f"Complete history response:\n{response}")
    
    # Count how many emails are mentioned
    email_count = 0
    if "alice" in response.lower() or "project alpha" in response.lower():
        email_count += 1
        print("✅ Email 1 (Alice/Project Alpha) found")
    
    if "bob" in response.lower() or "invoice" in response.lower():
        email_count += 1
        print("✅ Email 2 (Bob/Invoice) found")
    
    if "carol" in response.lower() or "meeting" in response.lower():
        email_count += 1
        print("✅ Email 3 (Carol/Meeting) found")
    
    print(f"\nTotal emails found in history: {email_count}/3")
    
    # Test 7: Access draft from previous session
    print("\n9. Testing access to previous session draft...")
    response = agent.process_user_input("show draft from email 2")
    print(f"Previous draft response:\n{response}")
    
    if "bob" in response.lower() or "invoice" in response.lower() or "draft" in response.lower():
        print("✅ Previous session draft accessible")
    else:
        print("❌ Previous session draft NOT accessible")
    
    print("\n" + "=" * 50)
    print("Session History Test Complete!")

if __name__ == "__main__":
    test_session_history()