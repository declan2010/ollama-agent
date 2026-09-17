"""
Unit tests for the core security and command execution logic of Ollama‑WebChat.

These tests focus on the helper functions that decide whether a shell command
is safe to run and on the `read <file>` alias that prevents infinite loops.
Run them with ``pytest``:

    pytest -q

The tests import the implementation module directly (``ollama_webchat.impl``)
so they need the project root to be on ``sys.path``.  The ``conftest.py``
file at the repo root adds the project directory automatically.
"""

import os
import sys

# Make sure the project root is importable when the tests are run from the
# repository root or from any sub‑directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ollama_webchat.impl import is_write_command, validate_command, already_read, remember_read


def test_read_alias_is_not_a_write():
    """The `read` command should never be classified as a write operation."""
    assert not is_write_command('read index.html')
    assert not is_write_command('read /home/user/file.txt')


def test_redirection_is_write():
    """Any command that redirects output (>, >>) is a write."""
    assert is_write_command('echo hello > out.txt')
    assert is_write_command('printf "x" >> log.txt')
    assert is_write_command('cat file | tee copy.txt')


def test_explicit_write_commands():
    """Commands listed in WRITE_COMMANDS are recognised as writes."""
    for cmd in ['touch newfile', 'rm -rf /tmp/test', 'mv a b', 'cp src dst']:
        assert is_write_command(cmd), f'{cmd} should be considered a write'


def test_interpreter_one_liners_are_writes():
    """Interpreter one‑liners can modify the filesystem."""
    assert is_write_command('python -c "open(\'foo.txt\',\'w\').write(\'x\')"')
    assert is_write_command('python3 -c "import os; os.remove(\'x\')"')
    assert is_write_command('node -e "require(\'fs\').writeFileSync(\'x\',\'y\')"')
    assert is_write_command('perl -e "open(F,\">x\"); print F 1; close F"')


def test_download_commands_are_writes():
    """curl/wget with an output flag write a file."""
    assert is_write_command('curl https://example.com -o file.html')
    assert is_write_command('wget https://example.com -O file.html')


def test_validate_command_whitelists():
    """The whitelist validator should accept known read‑only commands."""
    assert validate_command('ls -la') is not None
    assert validate_command('cat ollama_webchat/impl.py') is not None
    # The validator should reject unknown programs.
    assert validate_command('foobar --help') is None


def test_read_alias_lru():
    """`remember_read` should prevent duplicate reads after the first one."""
    path = os.path.abspath('dummy.txt')
    # First read: remembers
    assert remember_read(path) is True
    # Second read: already known
    assert remember_read(path) is False
    assert already_read(path) is True
    # Different path is not known yet
    assert already_read(os.path.abspath('other.txt')) is False
