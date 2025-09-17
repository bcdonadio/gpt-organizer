"""Tests for the ``debug_config`` helper module."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace, TracebackType
from typing import Any

import pytest

if "debugpy" not in sys.modules:
    stub = ModuleType("debugpy")

    def _noop_configure(**kwargs: Any) -> None:
        return None

    def _noop_listen(*args: Any, **kwargs: Any) -> None:
        return None

    def _noop_wait() -> None:
        return None

    def _noop_breakpoint() -> None:
        return None

    def _noop_is_client_connected() -> bool:
        return False

    setattr(stub, "configure", _noop_configure)
    setattr(stub, "listen", _noop_listen)
    setattr(stub, "wait_for_client", _noop_wait)
    setattr(stub, "breakpoint", _noop_breakpoint)
    setattr(stub, "is_client_connected", _noop_is_client_connected)
    sys.modules["debugpy"] = stub

import debug_config


class DebugpyStub:
    def __init__(self) -> None:
        self.actions: list[tuple[str, object]] = []
        self.connected: bool | Exception = False
        self.listen_side_effect: Exception | None = None

    def configure(self, **kwargs: object) -> None:
        self.actions.append(("configure", dict(kwargs)))

    def listen(self, addr: object) -> None:
        if self.listen_side_effect is not None:
            raise self.listen_side_effect
        if isinstance(addr, Exception):
            raise addr
        self.actions.append(("listen", addr))

    def wait_for_client(self) -> None:
        self.actions.append(("wait", None))

    def breakpoint(self) -> None:  # pragma: no cover - defensive
        self.actions.append(("breakpoint", None))

    def is_client_connected(self) -> bool:
        self.actions.append(("is_client_connected", None))
        if isinstance(self.connected, Exception):
            raise self.connected
        return bool(self.connected)


@pytest.fixture
def debugpy_stub(monkeypatch: pytest.MonkeyPatch) -> DebugpyStub:
    stub = DebugpyStub()
    monkeypatch.setattr(debug_config, "debugpy", stub)
    return stub


def test_start_debug_server_waits_for_client(debugpy_stub: DebugpyStub, capsys: pytest.CaptureFixture[str]) -> None:
    """The server should configure, listen, and optionally wait."""

    debug_config.start_debug_server(wait_for_client=True, log_to_stderr=True)
    out = capsys.readouterr().out
    assert "Debug server started" in out
    assert debugpy_stub.actions[0][0] == "configure"
    assert any(action[0] == "wait" for action in debugpy_stub.actions)


def test_start_debug_server_handles_exceptions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Failures during setup should be reported without raising."""

    stub = DebugpyStub()
    stub.listen_side_effect = RuntimeError("boom")
    monkeypatch.setattr(debug_config, "debugpy", stub)

    debug_config.start_debug_server()
    out = capsys.readouterr().out
    assert "Failed to start debug server" in out


def test_enable_debugging_on_exception_invokes_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unhandled exceptions should trigger the debug server."""

    called: list[tuple[tuple[str, object], ...]] = []

    def fake_start(**kwargs: object) -> None:
        called.append(tuple(kwargs.items()))

    monkeypatch.setattr(debug_config, "start_debug_server", fake_start)

    recorded: dict[str, tuple[type[BaseException], BaseException, TracebackType | None]] = {}

    def fake_excepthook(exc_type: type[BaseException], exc_value: BaseException, exc_tb: TracebackType | None) -> None:
        recorded["exc"] = (exc_type, exc_value, exc_tb)

    monkeypatch.setattr(sys, "__excepthook__", fake_excepthook)

    debug_config.enable_debugging_on_exception()
    sys.excepthook(RuntimeError, RuntimeError("fail"), None)

    assert called and called[0][0][0] == "wait_for_client"
    recorded_exc = recorded["exc"]
    assert recorded_exc[0] is RuntimeError


def test_enable_debugging_on_exception_skips_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """KeyboardInterrupt should not start debugging."""

    def fail_start(**kwargs: object) -> None:
        raise AssertionError()

    monkeypatch.setattr(debug_config, "start_debug_server", fail_start)
    debug_config.enable_debugging_on_exception()
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)


def test_debug_here_starts_and_breaks(debugpy_stub: DebugpyStub, capsys: pytest.CaptureFixture[str]) -> None:
    """``debug_here`` should block until a client connects and trigger a breakpoint."""

    debug_config.debug_here()
    out = capsys.readouterr().out
    assert "Waiting for debugger" in out
    assert ("wait", None) in debugpy_stub.actions
    assert ("breakpoint", None) in debugpy_stub.actions


def test_is_debugger_attached_reports_state(debugpy_stub: DebugpyStub) -> None:
    """``is_debugger_attached`` should proxy to debugpy."""

    debugpy_stub.connected = True
    assert debug_config.is_debugger_attached() is True
    debugpy_stub.connected = Exception("boom")
    assert debug_config.is_debugger_attached() is False


def test_auto_start_if_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment flags should trigger auto start with configured values."""

    captured: dict[str, object] = {}

    def fake_start(host: str, port: int, wait_for_client: bool) -> None:
        captured.update({"host": host, "port": port, "wait": wait_for_client})

    monkeypatch.setattr(debug_config, "start_debug_server", fake_start)
    monkeypatch.setenv("DEBUGPY_ENABLE", "true")
    monkeypatch.setenv("DEBUGPY_HOST", "0.0.0.0")
    monkeypatch.setenv("DEBUGPY_PORT", "9999")
    monkeypatch.setenv("DEBUGPY_WAIT", "yes")

    debug_config.auto_start_if_enabled()
    assert captured == {"host": "0.0.0.0", "port": 9999, "wait": True}


def test_auto_start_if_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the environment flag is not set nothing happens."""

    monkeypatch.delenv("DEBUGPY_ENABLE", raising=False)

    def fail_start(**kwargs: object) -> None:
        raise AssertionError()

    monkeypatch.setattr(debug_config, "start_debug_server", fail_start)
    debug_config.auto_start_if_enabled()


def test_debug_config_main_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running the module as a script should honor CLI arguments."""

    stub = DebugpyStub()
    monkeypatch.setitem(sys.modules, "debugpy", stub)
    monkeypatch.setenv("DEBUGPY_ENABLE", "0")

    argv: list[str] = ["debug_config", "--host", "127.0.0.1", "--port", "5679", "--wait", "--log"]
    monkeypatch.setattr(sys, "argv", argv)

    # Ensure the sleep loop exits immediately via a KeyboardInterrupt
    def raise_keyboard_interrupt(_: float) -> None:
        raise KeyboardInterrupt()

    fake_time = SimpleNamespace(sleep=raise_keyboard_interrupt)
    monkeypatch.setitem(sys.modules, "time", fake_time)

    runpy.run_module("debug_config", run_name="__main__")

    output = capsys.readouterr().out
    assert "Debug server running" in output
    assert ("listen", ("127.0.0.1", 5679)) in stub.actions
