"""Keep terminal-aware package progress visible while retaining raw command logs."""
from __future__ import annotations

import errno
import os
from pathlib import Path
import re
import shutil


class TerminalText:
    """Incrementally remove terminal escape sequences, including split CSI/OSC."""

    def __init__(self):
        self.state = 'text'
        self.last_line = ''
        self.line = ''

    def feed(self, text: str) -> str:
        result = []
        for char in text:
            if self.state == 'escape':
                self.state = 'csi' if char == '[' else 'osc' if char == ']' else 'text'
            elif self.state == 'csi':
                if '@' <= char <= '~':
                    if char in 'ABEFHfJK':
                        # Cursor motion/erasure separates redraws even without LF.
                        if self.line.strip():
                            self.last_line = self.line.strip()[-500:]
                            result.append('\n')
                        self.line = ''
                    self.state = 'text'
            elif self.state == 'osc':
                if char == '\x07':
                    self.state = 'text'
                elif char == '\x1b':
                    self.state = 'osc-escape'
            elif self.state == 'osc-escape':
                self.state = 'text' if char == '\\' else 'osc'
            elif char == '\x1b':
                self.state = 'escape'
            elif char in '\r\n':
                if self.line.strip():
                    self.last_line = self.line.strip()[-500:]
                self.line = ''
                result.append('\n')
            elif char == '\b':
                self.line = self.line[:-1]
            elif char >= ' ' or char == '\t':
                self.line = (self.line + char)[-1000:]
                result.append(char)
        return ''.join(result)

    @property
    def latest(self) -> str:
        value = self.line.strip()[-500:] or self.last_line
        # uv pads rows to the terminal width instead of emitting a newline.
        # Keep the final visible row and collapse padding in the summary only.
        return ' '.join(re.split(r' {32,}', value.strip())[-1].split())


class CommandOutput:
    """Use a PTY for uv progress on POSIX; keep other command output byte-for-byte."""

    def __init__(self, argv: list[str], log):
        self.log = log
        self.stdout = log
        self.master = self.slave = None
        self.mode = 'file'
        self.fallback = None
        executable = Path(argv[0]).name.lower() if argv else ''
        self.requested = executable in ('uv', 'uv.exe') and (
            'venv' in argv[1:3] or 'run' in argv[1:3] or
            ('pip' in argv[1:3] and 'install' in argv[2:4]))

    def __enter__(self):
        if not self.requested:
            return self
        if os.name != 'posix':
            self.fallback = '此平台不提供 POSIX PTY，使用实时文本输出。'
            return self
        try:
            import fcntl
            import pty
            import struct
            import termios
            self.master, self.slave = pty.openpty()
            attributes = termios.tcgetattr(self.slave)
            attributes[1] &= ~termios.OPOST  # Preserve LF and CR in the raw log.
            termios.tcsetattr(self.slave, termios.TCSANOW, attributes)
            columns = max(80, min(200, shutil.get_terminal_size((120, 30)).columns))
            fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack('HHHH', 30, columns, 0, 0))
            os.set_blocking(self.master, False)
            self.stdout, self.mode = self.slave, 'pty'
        except (ImportError, OSError) as error:
            self.close()
            self.fallback = f'PTY 不可用，使用实时文本输出：{error}'
        return self

    def spawned(self):
        if self.slave is not None:
            os.close(self.slave)
            self.slave = None

    def drain(self):
        if self.master is None:
            return
        # Bound each pass so noisy children cannot starve timeout/cancellation.
        for _ in range(16):
            try:
                chunk = os.read(self.master, 65536)
            except BlockingIOError:
                break
            except OSError as error:
                if error.errno == errno.EIO:  # Linux PTY EOF after the slave closes.
                    break
                raise
            if not chunk:
                break
            self.log.write(chunk)
        self.log.flush()

    def close(self):
        for name in ('slave', 'master'):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def __exit__(self, *exc):
        self.close()
