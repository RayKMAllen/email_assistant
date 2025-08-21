"""
Conversational CLI for the email assistant.
Provides a natural language interface instead of command-based interaction.
"""

import click
from typing import Optional

from assistant.conversational_agent import ConversationalEmailAgent


# Global agent instance
agent: Optional[ConversationalEmailAgent] = None


def get_agent() -> ConversationalEmailAgent:
    """Get or create the global agent instance"""
    global agent
    if agent is None:
        agent = ConversationalEmailAgent()
    return agent


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Conversational Email Assistant - Your AI-powered email helper"""
    if ctx.invoked_subcommand is None:
        run_conversational_shell()


@cli.command()
def chat():
    """Start a conversational chat session with the email assistant"""
    run_conversational_shell()


@cli.command()
@click.argument("message", nargs=-1)
def ask(message):
    """Ask the assistant a question or give it a command"""
    if not message:
        click.echo("Please provide a message. Example: eassistant ask 'Help me with this email'")
        return
    
    message_text = " ".join(message)
    
    try:
        agent = get_agent()
        response = agent.process_user_input(message_text)
        click.echo(response)
    except KeyboardInterrupt:
        click.echo("👋 Goodbye!")
        # Handle keyboard interrupt gracefully
    except Exception as e:
        click.echo(f"⚠️ Error: {e}")
        # Don't re-raise the exception to avoid exit code 1


@cli.command()
def reset():
    """Reset the conversation and start fresh"""
    agent = get_agent()
    agent.reset_conversation()
    click.echo("✨ Conversation reset! Starting fresh.")
    click.echo(agent.get_greeting_message())


@cli.command()
def status():
    """Show current conversation status and statistics"""
    agent = get_agent()
    summary = agent.get_conversation_summary()
    
    click.echo("📊 Conversation Status:")
    click.echo(f"   Current State: {summary['conversation_state']}")
    click.echo(f"   Messages Exchanged: {summary['conversation_count']}")
    click.echo(f"   Successful Operations: {summary['successful_operations']}")
    click.echo(f"   Failed Operations: {summary['failed_operations']}")
    click.echo(f"   Email Loaded: {'✅' if summary['has_email_loaded'] else '❌'}")
    click.echo(f"   Draft Available: {'✅' if summary['has_draft'] else '❌'}")
    click.echo(f"   Draft Versions: {summary['draft_history_count']}")


@cli.command()
def help_commands():
    """Show available commands (for users who prefer command-style interaction)"""
    click.echo("🤖 Email Assistant Commands:")
    click.echo("")
    click.echo("💬 Natural Language Commands (recommended):")
    click.echo("   Just type naturally! Examples:")
    click.echo("   • 'Here's an email I need help with: [email content]'")
    click.echo("   • 'Draft a formal reply to this email'")
    click.echo("   • 'Make the draft more professional'")
    click.echo("   • 'Save this draft to a file'")
    click.echo("")
    click.echo("⚙️ CLI Commands:")
    click.echo("   eassistant                    - Start conversational mode")
    click.echo("   eassistant ask 'message'      - Send a single message")
    click.echo("   eassistant reset              - Reset conversation")
    click.echo("   eassistant status             - Show conversation status")
    click.echo("   eassistant help-commands      - Show this help")
    click.echo("")
    click.echo("💡 Tips:")
    click.echo("   • The assistant understands natural language")
    click.echo("   • It will guide you through the email workflow")
    click.echo("   • You can paste emails directly or provide file paths")
    click.echo("   • Type 'help' or 'exit' during conversation")


def run_conversational_shell():
    """Run the main conversational interface"""
    agent = get_agent()
    
    # Show greeting
    click.echo("=" * 60)
    click.echo("🤖 Conversational Email Assistant")
    click.echo("=" * 60)
    click.echo(agent.get_greeting_message())
    click.echo("")
    click.echo("💡 Tips:")
    click.echo("   • Just type naturally - I understand conversational language")
    click.echo("   • Type 'help' for assistance, 'status' for current state")
    click.echo("   • Type 'exit', 'quit', or press Ctrl+C to leave")
    click.echo("   • Type 'reset' to start a new conversation")
    click.echo("")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                click.echo("👋 Goodbye! Thanks for using the Email Assistant!")
                break
            
            elif user_input.lower() in ['help', '?']:
                show_conversational_help()
                continue
            
            elif user_input.lower() == 'status':
                show_status_in_conversation(agent)
                continue
            
            elif user_input.lower() == 'reset':
                agent.reset_conversation()
                click.echo("✨ Conversation reset!")
                click.echo(agent.get_greeting_message())
                continue
            
            elif user_input.lower() == 'clear':
                # Clear screen (works on most terminals)
                click.clear()
                continue
            
            # Process user input through conversational agent
            try:
                response = agent.process_user_input(user_input)
                click.echo(f"\n🤖 Assistant: {response}\n")
                
            except KeyboardInterrupt:
                click.echo("\n👋 Goodbye!")
                break
            except Exception as e:
                click.echo(f"\n⚠️ I encountered an error: {e}")
                click.echo("Let's try that again. What would you like me to help you with?\n")
        
        except KeyboardInterrupt:
            click.echo("\n👋 Goodbye!")
            break
        except EOFError:
            click.echo("\n👋 Goodbye!")
            break


def show_conversational_help():
    """Show help information during conversation"""
    click.echo("\n🆘 Help - What I Can Do:")
    click.echo("")
    click.echo("📧 Email Processing:")
    click.echo("   • 'Here's an email: [paste email content]'")
    click.echo("   • 'Process this file: /path/to/email.pdf'")
    click.echo("   • 'I have an email I need help with'")
    click.echo("")
    click.echo("🔍 Information Extraction:")
    click.echo("   • 'What are the key details?'")
    click.echo("   • 'Show me the summary'")
    click.echo("   • 'Who sent this email?'")
    click.echo("")
    click.echo("✍️ Reply Drafting:")
    click.echo("   • 'Draft a reply'")
    click.echo("   • 'Write a formal response'")
    click.echo("   • 'Help me respond to this email'")
    click.echo("")
    click.echo("🔧 Draft Refinement:")
    click.echo("   • 'Make it more professional'")
    click.echo("   • 'Add a meeting request'")
    click.echo("   • 'Make it shorter and more concise'")
    click.echo("")
    click.echo("💾 Saving:")
    click.echo("   • 'Save this draft'")
    click.echo("   • 'Export to a file'")
    click.echo("   • 'Save to cloud storage'")
    click.echo("")
    click.echo("🔧 Special Commands:")
    click.echo("   • 'help' - Show this help")
    click.echo("   • 'status' - Show conversation status")
    click.echo("   • 'reset' - Start a new conversation")
    click.echo("   • 'clear' - Clear the screen")
    click.echo("   • 'exit' - Leave the assistant")
    click.echo("")


def show_status_in_conversation(agent: ConversationalEmailAgent):
    """Show status information during conversation"""
    summary = agent.get_conversation_summary()
    
    click.echo("\n📊 Current Status:")
    click.echo(f"   🔄 State: {summary['conversation_state'].replace('_', ' ').title()}")
    click.echo(f"   💬 Messages: {summary['conversation_count']}")
    click.echo(f"   ✅ Successful: {summary['successful_operations']}")
    click.echo(f"   ❌ Failed: {summary['failed_operations']}")
    click.echo(f"   📧 Email: {'Loaded' if summary['has_email_loaded'] else 'Not loaded'}")
    click.echo(f"   📝 Draft: {'Available' if summary['has_draft'] else 'Not created'}")
    
    if summary['draft_history_count'] > 0:
        click.echo(f"   🔄 Draft versions: {summary['draft_history_count']}")
    
    click.echo("")


if __name__ == "__main__":
    cli()
