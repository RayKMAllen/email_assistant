import click

from assistant.llm_session import BedrockSession

session = BedrockSession()

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """A simple interactive To-Do CLI."""
    if ctx.invoked_subcommand is None:
        run_shell()


# @cli.command()
# @click.argument("task")
# def add(task):
#     """Add a new TASK to the list."""
#     tasks.append(task)
#     click.echo(f"✅ Added: {task}")


# @cli.command()
# def list():
#     """Show all tasks."""
#     if not tasks:
#         click.echo("📋 No tasks yet!")
#     else:
#         click.echo("\nYour tasks:")
#         for i, t in enumerate(tasks, start=1):
#             click.echo(f"{i}. {t}")


# @cli.command()
# @click.argument("index", type=int)
# def remove(index):
#     """Remove a task by its INDEX (1-based)."""
#     try:
#         removed = tasks.pop(index - 1)
#         click.echo(f"🗑️ Removed: {removed}")
#     except IndexError:
#         click.echo("❌ Invalid index.")

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
    except Exception as e:
        click.echo(f"⚠️ Error extracting key info: {e}")



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
            # if cmd in ("help", "?"):
            #     click.echo("Available commands: load, list, remove, exit")
            #     continue
            cli.main(args=cmd.split(), prog_name="eassistant", standalone_mode=False)
        except SystemExit:
            break
        except Exception as e:
            click.echo(f"⚠️ Error: {e}")