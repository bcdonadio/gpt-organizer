"""
Debug configuration module for manual debugging with debugpy.

This module provides utilities to enable remote debugging capabilities
for both manual use and programmatic integration.
"""

import debugpy  # type: ignore
import os
import sys
from types import TracebackType
from typing import Type


def start_debug_server(
    host: str = "localhost", port: int = 5678, wait_for_client: bool = False, log_to_stderr: bool = False
) -> None:
    """
    Start the debugpy debug server.

    Args:
        host: Host to bind the debug server to (default: localhost)
        port: Port to bind the debug server to (default: 5678)
        wait_for_client: Whether to wait for a client to attach before continuing
        log_to_stderr: Whether to log debug messages to stderr
    """
    try:
        # Configure debugpy
        if log_to_stderr:
            debugpy.configure(python=None, qt=None, subProcess=True)

        # Start the debug server
        debugpy.listen((host, port))
        print(f"Debug server started on {host}:{port}")

        if wait_for_client:
            print("Waiting for debugger to attach...")
            debugpy.wait_for_client()
            print("Debugger attached!")

    except Exception as e:
        print(f"Failed to start debug server: {e}")


def enable_debugging_on_exception() -> None:
    """
    Enable automatic debugging when an unhandled exception occurs.
    This will start the debug server and wait for a client when an exception happens.
    """

    def excepthook(
        exc_type: Type[BaseException], exc_value: BaseException, exc_traceback: TracebackType | None
    ) -> None:
        if exc_type is KeyboardInterrupt:
            # Don't debug keyboard interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        print(f"Unhandled exception: {exc_type.__name__}: {exc_value}")
        start_debug_server(wait_for_client=True)

        # Call the original excepthook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = excepthook


def debug_here(host: str = "localhost", port: int = 5678) -> None:
    """
    Start debugging at the current location in code.

    Args:
        host: Host to bind the debug server to
        port: Port to bind the debug server to
    """
    try:
        debugpy.listen((host, port))
        print(f"Debug server started on {host}:{port}")
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        debugpy.breakpoint()  # This will pause execution here
    except Exception as e:
        print(f"Failed to start debugging: {e}")


def is_debugger_attached() -> bool:
    """
    Check if a debugger is currently attached.

    Returns:
        True if debugger is attached, False otherwise
    """
    try:
        return bool(debugpy.is_client_connected())
    except Exception:
        return False


# Environment variable based auto-start
def auto_start_if_enabled() -> None:
    """
    Automatically start debug server if DEBUGPY_ENABLE environment variable is set.

    Environment variables:
        DEBUGPY_ENABLE: Set to '1', 'true', 'yes' to enable
        DEBUGPY_HOST: Host to bind to (default: localhost)
        DEBUGPY_PORT: Port to bind to (default: 5678)
        DEBUGPY_WAIT: Set to '1', 'true', 'yes' to wait for client
    """
    if os.environ.get("DEBUGPY_ENABLE", "").lower() in ("1", "true", "yes"):
        host = os.environ.get("DEBUGPY_HOST", "localhost")
        port = int(os.environ.get("DEBUGPY_PORT", "5678"))
        wait = os.environ.get("DEBUGPY_WAIT", "").lower() in ("1", "true", "yes")

        start_debug_server(host=host, port=port, wait_for_client=wait)


# Auto-start if environment variable is set
auto_start_if_enabled()


if __name__ == "__main__":
    # When run as a script, start the debug server
    import argparse

    parser = argparse.ArgumentParser(description="Start debugpy debug server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5678, help="Port to bind to")
    parser.add_argument("--wait", action="store_true", help="Wait for client to attach")
    parser.add_argument("--log", action="store_true", help="Log to stderr")

    args = parser.parse_args()

    start_debug_server(host=args.host, port=args.port, wait_for_client=args.wait, log_to_stderr=args.log)

    print("Debug server running. Press Ctrl+C to stop.")
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDebug server stopped.")
