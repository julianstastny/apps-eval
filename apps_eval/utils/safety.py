"""
Security utilities for safe code execution.
"""
import os
import sys
import resource
import platform
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any

# Remove loguru import and add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


def set_memory_limit(max_bytes: int) -> None:
    """Set the memory limit for the current process."""
    if platform.system() != "Darwin":  # Memory limits don't work well on macOS
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (max_bytes, max_bytes))
        if platform.system() != "Darwin":  # Stack limits are different on macOS
            resource.setrlimit(resource.RLIMIT_STACK, (max_bytes, max_bytes))


def disable_dangerous_functions() -> None:
    """Disable potentially dangerous built-in functions and modules."""
    import builtins
    
    # Save original open function
    original_open = builtins.open
    
    # Disable dangerous builtins
    DANGEROUS_BUILTINS = {
        'exit', 'quit', 'help',  # Interactive functions
        'open'  # File operations
    }
    for func in DANGEROUS_BUILTINS:
        if hasattr(builtins, func):
            setattr(builtins, func, None)
    
    # Disable dangerous os functions
    DANGEROUS_OS_FUNCTIONS = {
        'system', 'popen', 'popen2', 'popen3', 'popen4',  # Command execution
        'execl', 'execle', 'execlp', 'execlpe', 'execv', 'execve', 'execvp', 'execvpe',  # Process execution
        'fork', 'forkpty',  # Process creation
        'kill', 'killpg',  # Process termination
        'remove', 'unlink', 'rmdir', 'removedirs',  # File deletion
        'rename', 'renames', 'replace',  # File operations
        'setuid', 'setgid', 'chown', 'chmod',  # Permissions
        'chroot', 'fchdir', 'chdir',  # Directory operations
    }
    for func in DANGEROUS_OS_FUNCTIONS:
        if hasattr(os, func):
            setattr(os, func, None)
    
    # Disable dangerous modules
    DANGEROUS_MODULES = {
        'subprocess',  # Command execution
        'socket',     # Network access
        'requests',   # HTTP requests
        'urllib',     # URL handling
        'ftplib',     # FTP access
        'poplib',     # POP3 access
        'imaplib',    # IMAP access
        'telnetlib',  # Telnet access
        'smtplib',    # SMTP access
        'multiprocessing',  # Process management
        'threading',  # Thread management
        'ctypes',    # C interface
        '_winreg',   # Windows registry
        'winreg',    # Windows registry
        'msvcrt',    # Windows console
    }
    for module in DANGEROUS_MODULES:
        sys.modules[module] = None
    
    return original_open


def restore_open_function(original_open):
    """Restore the original open function."""
    import builtins
    builtins.open = original_open


@contextmanager
def secure_execution_environment(max_memory_bytes: Optional[int] = None):
    """
    Context manager that creates a secure execution environment.
    
    Args:
        max_memory_bytes: Maximum memory usage in bytes
    """
    # Save original environment
    original_env = dict(os.environ)
    
    try:
        # Set resource limits
        if max_memory_bytes is not None:
            set_memory_limit(max_memory_bytes)
        
        # Disable dangerous functionality and save original open
        original_open = disable_dangerous_functions()
        
        yield
        
    except Exception as e:
        logging.exception("Error in secure execution environment")
        raise SecurityError(f"Security violation: {str(e)}")
        
    finally:
        # Restore original environment and open function
        try:
            os.environ.clear()
            os.environ.update(original_env)
            restore_open_function(original_open)
        except Exception as e:
            logging.error(f"Error restoring environment: {e}")
            # Create a new environment dict if the original is corrupted
            os.environ = dict(original_env) 