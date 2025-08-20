# Conversational Email Assistant

**A context-aware conversational AI agent that helps you process emails and draft professional replies through natural language interaction.**

The Email Assistant is an intelligent, conversational agent accessible via command line interface (CLI) that understands natural language and proactively guides you through email processing workflows. Instead of memorizing commands, simply talk to it naturally!

## 🌟 Key Features

### 🤖 **True Conversational Interface**
- **Natural Language Understanding**: Just type naturally - "Here's an email I need help with" instead of learning commands
- **Context-Aware Responses**: Remembers your conversation and provides relevant suggestions
- **Proactive Guidance**: Automatically suggests next steps and guides you through the workflow
- **Intelligent Intent Recognition**: Understands what you want even with ambiguous requests

### 📧 **Comprehensive Email Processing**
- **Multi-Format Support**: Process emails from text, file paths, or PDF documents
- **Smart Information Extraction**: Automatically extracts sender, receiver, subject, and key details
- **Professional Reply Drafting**: Creates contextually appropriate responses
- **Flexible Tone Control**: Adjust replies to be formal, casual, professional, or friendly
- **Iterative Refinement**: Keep improving drafts until they're perfect

### 🔄 **Intelligent Workflow Management**
- **State-Aware Processing**: Tracks where you are in the email workflow
- **Seamless Transitions**: Smoothly moves from loading → extracting → drafting → refining → saving
- **Error Recovery**: Gracefully handles issues and guides you back on track
- **Multi-Email Support**: Process multiple emails in a single conversation

### ☁️ **Cloud Integration**
- **AWS Bedrock Integration**: Powered by Claude LLM for intelligent responses
- **S3 Storage Support**: Save drafts to cloud storage or local files
- **Secure Processing**: Your data is processed securely with AWS infrastructure

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RayKMAllen/email_assistant.git
   cd email_assistant
   ```

2. **Configure your settings:**
   Edit `config.config` in the root directory:
   ```ini
   model_id = eu.anthropic.claude-3-7-sonnet-20250219-v1:0
   bucket_name = your-s3-bucket-name
   ```

3. **Install the package:**
   ```bash
   pip install .
   ```

### Basic Usage

**Start a conversation:**
```bash
eassistant
```

**Quick single interactions:**
```bash
eassistant ask "Help me with this email"
```

## 💬 Conversational Examples

### Example 1: Complete Email Workflow
```
🤖 Assistant: Hello! I'm your email assistant. I can help you process emails, extract key information, and draft professional replies. What can I help you with today?

You: Here's an email I need to respond to: 
From: john.smith@company.com
To: me@mycompany.com
Subject: Quarterly Review Meeting

Hi there,
I hope you're doing well. I wanted to reach out about scheduling our Q4 review meeting. Would you be available next week? Let me know what works best for you.
Best regards, John

🤖 Assistant: I've processed your email from John Smith about the quarterly review meeting. Here's a quick summary: John is requesting to schedule the Q4 review meeting and asking for your availability next week.

Would you like me to extract the detailed key information and draft a reply for you?

You: Yes, please draft a professional reply

🤖 Assistant: I've drafted a professional reply for you:

Subject: Re: Quarterly Review Meeting

Hi John,

Thank you for reaching out about the Q4 review meeting. I'd be happy to schedule this with you.

I'm available next week on Tuesday afternoon, Wednesday morning, or Thursday afternoon. Please let me know which of these times works best for you, and I'll send out a calendar invitation.

Looking forward to our discussion.

Best regards,
[Your name]

How does this look? I can refine it to be more formal, concise, friendly, or make any other adjustments you'd like. Or if you're happy with it, I can save it for you.

You: Make it more concise and add that I prefer Tuesday

🤖 Assistant: I've made the draft more concise and included your Tuesday preference:

Subject: Re: Quarterly Review Meeting

Hi John,

Thanks for reaching out. I'm available for the Q4 review meeting next week, with Tuesday afternoon being my preference. 

Please let me know if Tuesday works for you, or if you'd prefer Wednesday morning or Thursday afternoon instead.

Best regards,
[Your name]

Is this better? Would you like me to make any other changes, or shall I save this version for you?

You: Perfect, save it

🤖 Assistant: Great! I've saved your draft to 'draft_20250820_174500.txt'. Is there anything else I can help you with? I can process another email or assist with any other email-related tasks.
```

### Example 2: Quick Help and Clarification
```
You: I need help with something

🤖 Assistant: I'd be happy to help! Could you clarify what you'd like me to do? For example, you could:
- Share an email you'd like me to process
- Ask me what I can do
- Provide a file path to an email document

What works best for you?

You: What can you do?

🤖 Assistant: I'm your email assistant! Here's what I can help you with:

📧 **Process emails** - Load from text, file paths, or PDF files
🔍 **Extract key information** - Get sender, receiver, subject, and summary
✍️ **Draft replies** - Create professional responses with customizable tone
🔧 **Refine drafts** - Make them more formal, casual, concise, or add specific content
💾 **Save drafts** - Export to local files or cloud storage
🔄 **Iterative refinement** - Keep improving until you're satisfied

I can help you process emails, extract key information, and draft professional replies. You can paste an email directly, provide a file path, or ask me about my capabilities!
```

## 🛠️ Available Commands

### Conversational Mode (Recommended)
- **`eassistant`** - Start interactive conversation
- **`eassistant chat`** - Alternative way to start conversation

### Quick Commands
- **`eassistant ask "message"`** - Send a single message
- **`eassistant reset`** - Reset conversation and start fresh
- **`eassistant status`** - Show current conversation status
- **`eassistant help-commands`** - Show all available commands

### During Conversation
- **`help`** - Show what the assistant can do
- **`status`** - Check current conversation state
- **`reset`** - Start a new conversation
- **`clear`** - Clear the screen
- **`exit`** - Leave the assistant

## 🎯 Natural Language Patterns

The assistant understands many ways to express your needs:

### Loading Emails
- "Here's an email I need help with: [content]"
- "Process this file: /path/to/email.pdf"
- "I have an email from John about the meeting"
- "Can you help me with this email?"

### Drafting Replies
- "Draft a reply"
- "Help me respond to this"
- "Write a formal response"
- "I need to reply professionally"

### Refining Drafts
- "Make it more professional"
- "Add a meeting request"
- "Make it shorter and more polite"
- "Change the tone to be more casual"

### Getting Information
- "What are the key details?"
- "Show me the summary"
- "Who sent this email?"
- "Extract the important information"

### Saving Work
- "Save this draft"
- "Export to a file"
- "Keep this version"
- "Save to cloud storage"

## 📋 Requirements

- **Python 3.8+**
- **AWS credentials** with access to Bedrock
- **Dependencies**: click, PyMuPDF, boto3

## 🏗️ Architecture

The assistant uses a sophisticated conversational architecture:

- **Hybrid Intent Classification**: Rule-based patterns + LLM intelligence
- **Conversation State Management**: Tracks workflow progress and context
- **Proactive Response Generation**: Suggests next steps automatically
- **Error Recovery**: Graceful handling of issues with helpful guidance

See [`Architecture.md`](Architecture.md) for detailed technical information.

## 🔧 Configuration

### AWS Setup
1. Configure AWS credentials (via AWS CLI, environment variables, or IAM roles)
2. Ensure access to AWS Bedrock service
3. Set up S3 bucket for cloud storage (optional)

### Model Configuration
Edit `config.config`:
```ini
model_id = eu.anthropic.claude-3-7-sonnet-20250219-v1:0
bucket_name = your-s3-bucket-name
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest
```

## 📁 Project Structure

```
email_assistant/
├── src/
│   ├── assistant/
│   │   ├── conversational_agent.py    # Main conversational agent
│   │   ├── intent_classifier.py       # Hybrid intent classification
│   │   ├── conversation_state.py      # State management
│   │   ├── response_generator.py      # Natural language responses
│   │   ├── llm_session.py            # LLM integration
│   │   ├── prompts_params.py         # Prompts and parameters
│   │   └── utils.py                  # Utility functions
│   └── cli/
│       └── cli.py                    # Conversational CLI interface
├── tests/                            # Test suite
├── Architecture.md                   # Technical architecture
├── config.config                     # Configuration file
└── README.md                        # This file
```

## 🤝 Contributing

Contributions are welcome! The conversational architecture is designed to be extensible:

- **Add new intents** in [`intent_classifier.py`](src/assistant/intent_classifier.py)
- **Extend conversation states** in [`conversation_state.py`](src/assistant/conversation_state.py)
- **Improve response templates** in [`response_generator.py`](src/assistant/response_generator.py)
- **Enhance email processing** in [`llm_session.py`](src/assistant/llm_session.py)

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Raymond Allen** - [GitHub](https://github.com/RayKMAllen)

---

**Transform your email workflow with intelligent conversation!** 🚀
