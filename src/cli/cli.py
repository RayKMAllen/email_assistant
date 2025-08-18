import click

from assistant.llm_session import BedrockSession

session = BedrockSession()

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """A simple interactive To-Do CLI."""
    if ctx.invoked_subcommand is None:
        run_shell()

@cli.command()
@click.argument("path_or_text", type=str, nargs=-1)
def load(path_or_text):
    """Load an email from a file path or raw text."""

    # combine arguments into a single string
    path_or_text = " ".join(path_or_text)

    try:
        session.load_text(path_or_text)
        click.echo("Email content loaded successfully.")
    except Exception as e:
        click.echo(f"⚠️ Error loading email: {e}")
    try:
        session.extract_key_info()
        click.echo("Type 'draft' to draft a reply, with an optional tone argument (e.g. 'draft formal').")
    except Exception as e:
        click.echo(f"⚠️ Error extracting key info: {e}")

@cli.command()
@click.argument("tone", type=str, required=False)
def draft(tone):
    """Draft a reply to the loaded email."""
    if session.text is None or session.key_info is None:
        click.echo("⚠️ Correct email information has not yet been loaded. Please use the 'load' command first.")
        return
    try:
        reply = session.draft_reply(tone=tone)
        click.echo("Drafted reply:\n")
        click.echo(reply)
        click.echo("Type 'save' to save the draft to a file.")
        click.echo("Type 'refine' with additional instructions to refine the draft.")
    except Exception as e:
        click.echo(f"⚠️ Error drafting reply: {e}")

@cli.command()
@click.argument("filepath", type=click.Path(writable=True), required=False)
def save(filepath=None):
    """Save the drafted reply to a file."""
    if session.last_draft is None:
        click.echo("⚠️ No draft available. Please use the 'draft' command first.")
        return
    try:
        session.save_draft(filepath)
        click.echo("Draft saved successfully." if filepath else "Draft saved to default location.")
    except Exception as e:
        click.echo(f"⚠️ Error saving draft: {e}")

@cli.command()
@click.argument("instructions", type=str, nargs=-1)
def refine(instructions):
    """Refine the drafted reply with additional instructions."""
    if session.last_draft is None:
        click.echo("⚠️ No draft available. Please use the 'draft' command first.")
        return
    instructions = " ".join(instructions)
    try:
        refined_reply = session.refine(instructions)
        click.echo("Refined reply:\n")
        click.echo(refined_reply)
        click.echo("Type 'save' to save the refined draft to a file.")
        click.echo("Type 'refine' with additional instructions to further refine.")
    except Exception as e:
        click.echo(f"⚠️ Error refining reply: {e}")

@cli.command()
def exit():
    """Exit the CLI."""
    click.echo("👋 Goodbye!")
    raise SystemExit


def run_shell():
    """Simple REPL loop for interactive mode."""
    # click.echo("Welcome to the To-Do CLI! Type 'help' to see commands.")
    click.echo("Welcome! To start, please load the email path or content.")
    while True:
        try:
            cmd = input("> ").strip()
            if not cmd:
                continue
            if cmd in ("help", "?"):
                click.echo("Available commands: load, draft, refine, save, exit")
                continue
            cli.main(args=cmd.split(), prog_name="eassistant", standalone_mode=False)
        except SystemExit:
            break
        except Exception as e:
            click.echo(f"⚠️ Error: {e}")
