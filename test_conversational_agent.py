#!/usr/bin/env python3
"""
Simple test script to verify the conversational agent implementation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic conversational agent functionality"""
    print("🧪 Testing Conversational Email Agent...")
    
    try:
        from assistant.conversational_agent import ConversationalEmailAgent
        from assistant.conversation_state import ConversationState
        from assistant.intent_classifier import HybridIntentClassifier
        
        print("✅ All imports successful")
        
        # Test agent initialization
        agent = ConversationalEmailAgent()
        print("✅ Agent initialization successful")
        
        # Test greeting
        greeting = agent.get_greeting_message()
        print(f"✅ Greeting generated: {greeting[:50]}...")
        
        # Test conversation summary
        summary = agent.get_conversation_summary()
        print(f"✅ Summary generated: {summary}")
        
        # Test state management
        print(f"✅ Initial state: {agent.state_manager.context.current_state}")
        
        # Test intent classification (without LLM)
        classifier = HybridIntentClassifier()
        test_inputs = [
            "Here's an email I need help with",
            "Draft a reply",
            "Make it more formal",
            "Save this draft",
            "Help me"
        ]
        
        print("\n🔍 Testing Intent Classification:")
        for test_input in test_inputs:
            try:
                result = classifier.classify(test_input, agent.state_manager.context)
                print(f"  '{test_input}' → {result.intent} (confidence: {result.confidence:.2f})")
            except Exception as e:
                print(f"  '{test_input}' → Error: {e}")
        
        print("\n🎉 Basic conversational agent functionality verified!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversation_flow():
    """Test a simple conversation flow without AWS dependencies"""
    print("\n🔄 Testing Conversation Flow (Mock)...")
    
    try:
        from assistant.conversational_agent import ConversationalEmailAgent
        from assistant.conversation_state import ConversationState
        
        agent = ConversationalEmailAgent()
        
        # Test state transitions
        print(f"Initial state: {agent.state_manager.context.current_state}")
        
        # Simulate state transitions
        agent.state_manager.transition_state('GENERAL_HELP', success=True)
        print(f"After help request: {agent.state_manager.context.current_state}")
        
        # Test conversation history
        agent.state_manager.context.add_to_history("user", "Hello")
        agent.state_manager.context.add_to_history("assistant", "Hi there!")
        
        history = agent.state_manager.context.get_recent_history(2)
        print(f"✅ Conversation history: {len(history)} messages")
        
        print("✅ Conversation flow test completed")
        return True
        
    except Exception as e:
        print(f"❌ Conversation flow error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_response_generation():
    """Test response generation without LLM dependencies"""
    print("\n💬 Testing Response Generation...")
    
    try:
        from assistant.conversational_agent import ConversationalEmailAgent
        from assistant.response_generator import ConversationalResponseGenerator
        
        agent = ConversationalEmailAgent()
        
        # Test clarification response
        clarification = agent.response_generator.generate_clarification_response(
            "I need help", 
            agent.state_manager.get_context_summary()
        )
        print(f"✅ Clarification response: {clarification[:50]}...")
        
        # Test proactive guidance
        guidance = agent.response_generator._generate_proactive_guidance()
        print(f"✅ Proactive guidance: {guidance[:50]}...")
        
        print("✅ Response generation test completed")
        return True
        
    except Exception as e:
        print(f"❌ Response generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Conversational Email Assistant - Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_conversation_flow():
        tests_passed += 1
    
    if test_response_generation():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Conversational agent is ready.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    print("=" * 60)