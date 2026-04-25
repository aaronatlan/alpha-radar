"""Tests du `EmailNotifier` (Phase 3 étape 4)."""
from __future__ import annotations

import smtplib
from types import SimpleNamespace

import pytest

from alerts import notifier as notifier_mod
from alerts.notifier import EmailNotifier, SMTPConfig, format_subject


# ------------------------------------------------------ format_subject


def test_format_subject_uppercases_severity():
    s = format_subject("info", "new_thesis", "NVDA score 80")
    assert s.startswith("[ALPHA RADAR][INFO][new_thesis]")
    assert "NVDA score 80" in s


def test_format_subject_critical():
    s = format_subject("critical", "sector_heat_surge", "ai_ml +25")
    assert "[CRITICAL]" in s


# --------------------------------------------------- SMTPConfig.from_settings


def test_smtp_config_none_when_incomplete(monkeypatch):
    """Si l'un des champs requis manque, from_settings renvoie None."""
    stub = SimpleNamespace(
        smtp_host=None,            # manquant
        smtp_user="u",
        smtp_password="p",
        smtp_to="t@x",
        smtp_port=587,
        smtp_from=None,
        smtp_use_tls=True,
    )
    monkeypatch.setattr(notifier_mod, "settings", stub)
    assert SMTPConfig.from_settings() is None


def test_smtp_config_built_when_complete(monkeypatch):
    stub = SimpleNamespace(
        smtp_host="smtp.example.com",
        smtp_user="user@x",
        smtp_password="pwd",
        smtp_to="dest@x",
        smtp_port=465,
        smtp_from=None,             # → fallback sur user
        smtp_use_tls=False,
    )
    monkeypatch.setattr(notifier_mod, "settings", stub)
    cfg = SMTPConfig.from_settings()
    assert cfg is not None
    assert cfg.host == "smtp.example.com"
    assert cfg.port == 465
    assert cfg.sender == "user@x"
    assert cfg.recipient == "dest@x"
    assert cfg.use_tls is False


def test_smtp_config_uses_explicit_from(monkeypatch):
    stub = SimpleNamespace(
        smtp_host="h", smtp_user="u", smtp_password="p", smtp_to="t",
        smtp_port=587, smtp_from="bot@example.com", smtp_use_tls=True,
    )
    monkeypatch.setattr(notifier_mod, "settings", stub)
    assert SMTPConfig.from_settings().sender == "bot@example.com"


# ------------------------------------------------------ EmailNotifier (no-op)


def test_notifier_disabled_when_config_missing(monkeypatch):
    stub = SimpleNamespace(
        smtp_host=None, smtp_user=None, smtp_password=None, smtp_to=None,
        smtp_port=587, smtp_from=None, smtp_use_tls=True,
    )
    monkeypatch.setattr(notifier_mod, "settings", stub)
    n = EmailNotifier()
    assert n.is_enabled is False
    # send doit retourner False sans planter.
    assert n.send(subject="s", body="b") is False


# ---------------------------------------------- EmailNotifier — TLS happy path


class _FakeSMTP:
    """Stub minimaliste capturant les appels critiques."""

    instances: list["_FakeSMTP"] = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.starttls_called = False
        self.logged_in_with: tuple[str, str] | None = None
        self.sent_messages: list = []
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.starttls_called = True

    def login(self, user, password):
        self.logged_in_with = (user, password)

    def send_message(self, msg):
        self.sent_messages.append(msg)


def test_notifier_sends_via_starttls(monkeypatch):
    cfg = SMTPConfig(
        host="smtp.example.com", port=587,
        user="u@x", password="pwd",
        sender="u@x", recipient="dest@x", use_tls=True,
    )
    _FakeSMTP.instances.clear()
    monkeypatch.setattr(notifier_mod.smtplib, "SMTP", _FakeSMTP)

    n = EmailNotifier(config=cfg)
    assert n.is_enabled is True
    ok = n.send(subject="hello", body="world")
    assert ok is True
    inst = _FakeSMTP.instances[0]
    assert inst.starttls_called is True
    assert inst.logged_in_with == ("u@x", "pwd")
    assert len(inst.sent_messages) == 1
    msg = inst.sent_messages[0]
    assert msg["Subject"] == "hello"
    assert msg["From"] == "u@x"
    assert msg["To"] == "dest@x"


def test_notifier_uses_ssl_when_tls_disabled(monkeypatch):
    cfg = SMTPConfig(
        host="smtp.example.com", port=465,
        user="u@x", password="pwd",
        sender="u@x", recipient="dest@x", use_tls=False,
    )
    _FakeSMTP.instances.clear()
    monkeypatch.setattr(notifier_mod.smtplib, "SMTP_SSL", _FakeSMTP)

    n = EmailNotifier(config=cfg)
    ok = n.send(subject="hello", body="world")
    assert ok is True
    inst = _FakeSMTP.instances[0]
    # En mode SSL natif, pas de starttls.
    assert inst.starttls_called is False
    assert inst.logged_in_with == ("u@x", "pwd")


def test_notifier_returns_false_on_smtp_error(monkeypatch):
    cfg = SMTPConfig(
        host="smtp.example.com", port=587,
        user="u@x", password="pwd",
        sender="u@x", recipient="dest@x", use_tls=True,
    )

    class _BoomSMTP:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def starttls(self): raise smtplib.SMTPException("boom")
        def login(self, *a): pass
        def send_message(self, msg): pass

    monkeypatch.setattr(notifier_mod.smtplib, "SMTP", _BoomSMTP)
    n = EmailNotifier(config=cfg)
    # Ne lève pas — log et renvoie False.
    assert n.send(subject="s", body="b") is False


def test_notifier_returns_false_on_oserror(monkeypatch):
    """Une erreur réseau (OSError, ConnectionRefusedError…) doit être avalée."""
    cfg = SMTPConfig(
        host="smtp.example.com", port=587,
        user="u@x", password="pwd",
        sender="u@x", recipient="dest@x", use_tls=True,
    )

    def _raises(*a, **kw):
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(notifier_mod.smtplib, "SMTP", _raises)
    n = EmailNotifier(config=cfg)
    assert n.send(subject="s", body="b") is False


def test_notifier_explicit_config_bypasses_settings(monkeypatch):
    """Quand on passe un config explicite, settings n'est pas consulté."""
    # Force settings à une config invalide — l'explicite doit gagner.
    stub = SimpleNamespace(
        smtp_host=None, smtp_user=None, smtp_password=None, smtp_to=None,
        smtp_port=587, smtp_from=None, smtp_use_tls=True,
    )
    monkeypatch.setattr(notifier_mod, "settings", stub)

    cfg = SMTPConfig(
        host="h", port=587, user="u", password="p",
        sender="u", recipient="t", use_tls=True,
    )
    n = EmailNotifier(config=cfg)
    assert n.is_enabled is True
