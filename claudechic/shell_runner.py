"""PTY-based shell command execution with color support."""

import os
import pty
import select
import subprocess


def run_in_pty(
    cmd: str, shell: str, cwd: str | None, env: dict[str, str]
) -> tuple[str, int]:
    """Run command in PTY to capture colors.

    Returns (output, returncode) tuple.
    """
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            [shell, "-lc", cmd],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=cwd,
            env=env,
            close_fds=True,
            start_new_session=True,
        )
        os.close(slave_fd)

        output = b""
        while True:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if r:
                try:
                    data = os.read(master_fd, 4096)
                    if data:
                        output += data
                    else:
                        break
                except OSError:
                    break
            elif proc.poll() is not None:
                # Process done, drain remaining output
                while True:
                    r, _, _ = select.select([master_fd], [], [], 0.05)
                    if not r:
                        break
                    try:
                        data = os.read(master_fd, 4096)
                        if data:
                            output += data
                        else:
                            break
                    except OSError:
                        break
                break

        os.close(master_fd)
        proc.wait()
        return output.decode(errors="replace"), proc.returncode or 0
    except Exception:
        os.close(master_fd)
        raise
