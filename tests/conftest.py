"""Fixtures partagées à tous les tests.

`tmp_db` : crée une base SQLite vierge sur disque temporaire et
reconfigure les singletons du module `memory.database` pour qu'ils
pointent dessus le temps du test. Les tests n'écrivent jamais dans la
vraie base du projet.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from memory import database as db_mod


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch):
    """Base SQLite isolée pour un test."""
    db_file = tmp_path / "test.db"
    url = f"sqlite:///{db_file}"

    # Reset des singletons du module.
    db_mod._engine = None
    db_mod._SessionFactory = None

    # `settings.db_url` est une property — on ne peut pas setattr dessus.
    # On remplace donc l'objet `settings` utilisé par le module par un
    # stub simple qui expose `db_url`.
    stub_settings = SimpleNamespace(db_url=url)
    monkeypatch.setattr(db_mod, "settings", stub_settings)

    db_mod.init_db()
    yield url

    # Teardown : reset pour ne pas polluer les tests suivants.
    db_mod._engine = None
    db_mod._SessionFactory = None
