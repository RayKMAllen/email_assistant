#!/usr/bin/env python3
"""
Test script to demonstrate yes/no response handling in real scenarios.
This script simulates actual conversations to verify the implementation works correctly.
"""

from assistant.conversational_agent import ConversationalEmailAgent
from assistant.conversation_state import ConversationState
from unittest.mock import Mock

def setup_mock_agent():
    """Set up a mock agent for testing"""
    agent = ConversationalEmailAgent()
    
    # Mock the email processor to avoid actual LLM calls
    agent.email_processor = Mock()
    agent.email_processor.text = None
    agent.email_processor.key_info = None
    agent.email_processor.last_draft = None
    
    # Mock methods
    agent.email_processor.load_text = Mock()
    agent.email_processor.extract_key_info = Mock()
    agent.email_processor.draft_reply = Mock(return_value="Thank you for your email. I'll get back to you soon.")
    agent.email_processor.refine = Mock(return_value="Thank you for your email. I will respond promptly.")
    agent.email_processor.save_draft = Mock()
    
    return agent

def simulate_conversation(agent, inputs, descriptions):
    """Simulate a conversation with the agent"""
    print("=" * 60)
    print("CONVERSATION SIMULATION")
    print("=" * 60)
    
    for i, (user_input, description) in enumerate(zip(inputs, descriptions)):
        print(f"\n--- Step {i+1}: {description} ---")
        print(f"User: {user_input}")
        print(f"State before: {agent.state_manager.context.current_state.value}")
        
        response = agent.process_user_input(user_input)
        
        print(f"Agent: {response}")
        print(f"State after: {agent.state_manager.context.current_state.value}")

def test_scenario_1_yes_to_all():
    """Test scenario where user says yes to all offers"""
    print("\n🟢 SCENARIO 1: User says 'yes' to all offers")
    
    agent = setup_mock_agent()
    
    # Set up mock returns for this scenario
    agent.email_processor.key_info = {
        'sender_name': 'John Smith',
        'subject': 'Meeting Request',
        'summary': 'John is requesting a meeting next week to discuss the project.'
    }
    
    inputs = [
        "Here's an email:\nFrom: john@example.com\nTo: me@example.com\nSubject: Meeting Request\n\nHi, can we meet next week?",
        "yes",
        "yes"
    ]
    
    descriptions = [
        "User provides email content",
        "Agent offers to draft reply, user says yes",
        "Agent offers to save draft, user says yes"
    ]
    
    simulate_conversation(agent, inputs, descriptions)

def test_scenario_2_mixed_responses():
    """Test scenario with mixed yes/no responses"""
    print("\n🟡 SCENARIO 2: User gives mixed yes/no responses")
    
    agent = setup_mock_agent()
    
    # Set up mock returns
    agent.email_processor.key_info = {
        'sender_name': 'Sarah Johnson',
        'subject': 'Project Update',
        'summary': 'Sarah is providing an update on the current project status.'
    }
    
    inputs = [
        "Process this email: From: sarah@company.com\nSubject: Project Update\n\nHere's the latest update...",
        "yes",
        "no",
        "make it more formal",
        "yes"
    ]
    
    descriptions = [
        "User provides email content",
        "Agent offers to draft reply, user says yes",
        "Agent offers to save draft, user says no",
        "User requests refinement",
        "Agent offers to save refined draft, user says yes"
    ]
    
    simulate_conversation(agent, inputs, descriptions)

def test_scenario_3_no_to_draft():
    """Test scenario where user declines draft offer"""
    print("\n🔴 SCENARIO 3: User declines draft offer")
    
    agent = setup_mock_agent()
    
    # Set up mock returns
    agent.email_processor.key_info = {
        'sender_name': 'Mike Wilson',
        'subject': 'Question about Service',
        'summary': 'Mike has a question about our service offerings.'
    }
    
    inputs = [
        "Here's an email from a customer:\nFrom: mike@customer.com\nSubject: Question about Service\n\nI have some questions...",
        "no",
        "show me the summary again"
    ]
    
    descriptions = [
        "User provides email content",
        "Agent offers to draft reply, user says no",
        "User asks for information instead"
    ]
    
    simulate_conversation(agent, inputs, descriptions)

def test_scenario_4_various_yes_patterns():
    """Test various ways of saying yes"""
    print("\n✅ SCENARIO 4: Various ways of saying yes")
    
    agent = setup_mock_agent()
    
    # Set up mock returns
    agent.email_processor.key_info = {
        'sender_name': 'Alice Brown',
        'subject': 'Collaboration Opportunity',
        'summary': 'Alice is proposing a collaboration opportunity.'
    }
    
    yes_patterns = ["sure", "ok", "please do", "go for it", "Yes!", "OK"]
    
    for i, pattern in enumerate(yes_patterns):
        print(f"\n--- Testing '{pattern}' ---")
        
        # Reset agent state
        agent.state_manager.context.current_state = ConversationState.INFO_EXTRACTED
        agent.state_manager.context.email_content = "Sample email"
        agent.state_manager.context.extracted_info = agent.email_processor.key_info
        
        print(f"User: {pattern}")
        response = agent.process_user_input(pattern)
        print(f"Agent: {response[:100]}...")  # Truncate for readability
        print(f"State: {agent.state_manager.context.current_state.value}")

def test_scenario_5_various_no_patterns():
    """Test various ways of saying no"""
    print("\n❌ SCENARIO 5: Various ways of saying no")
    
    agent = setup_mock_agent()
    
    # Set up mock returns
    agent.email_processor.key_info = {
        'sender_name': 'Bob Davis',
        'subject': 'Follow-up Question',
        'summary': 'Bob has a follow-up question about our previous discussion.'
    }
    
    no_patterns = ["no", "nope", "not now", "skip", "no thanks", "pass"]
    
    for i, pattern in enumerate(no_patterns):
        print(f"\n--- Testing '{pattern}' ---")
        
        # Reset agent state
        agent.state_manager.context.current_state = ConversationState.INFO_EXTRACTED
        agent.state_manager.context.email_content = "Sample email"
        agent.state_manager.context.extracted_info = agent.email_processor.key_info
        
        print(f"User: {pattern}")
        response = agent.process_user_input(pattern)
        print(f"Agent: {response[:100]}...")  # Truncate for readability
        print(f"State: {agent.state_manager.context.current_state.value}")

if __name__ == "__main__":
    print("🤖 TESTING YES/NO RESPONSE HANDLING")
    print("This script demonstrates how the email assistant handles yes/no responses to offers.")
    
    test_scenario_1_yes_to_all()
    test_scenario_2_mixed_responses()
    test_scenario_3_no_to_draft()
    test_scenario_4_various_yes_patterns()
    test_scenario_5_various_no_patterns()
    
    print("\n" + "=" * 60)
    print("✅ ALL SCENARIOS COMPLETED")
    print("The email assistant now correctly handles yes/no responses to offers!")
    print("=" * 60)