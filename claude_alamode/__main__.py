"""Entry point for claude-alamode CLI."""

import argparse
import sys

from claude_alamode.app import ChatApp
from claude_alamode.errors import setup_logging

# Set up file logging to ~/claude-alamode.log
setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Claude à la Mode")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="Resume the most recent session"
    )
    parser.add_argument("--session", "-s", type=str, help="Resume a specific session ID")
    parser.add_argument("prompt", nargs="*", help="Initial prompt to send")
    args = parser.parse_args()

    initial_prompt = " ".join(args.prompt) if args.prompt else None

    # Pass resume flag or specific session ID - actual lookup happens in app
    resume_id = args.session if args.session else ("__most_recent__" if args.resume else None)

    # Set terminal window title
    from pathlib import Path
    sys.stdout.write(f"\033]0;Claude à la Mode · {Path.cwd().name}\007")
    sys.stdout.flush()

    try:
        app = ChatApp(resume_session_id=resume_id, initial_prompt=initial_prompt)
        app.run()
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        import traceback

        with open("/tmp/claude-alamode-crash.log", "w") as f:
            traceback.print_exc(file=f)
        raise


if __name__ == "__main__":
    main()
