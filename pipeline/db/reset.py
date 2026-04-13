"""Hard reset — deletes the SQLite database and LightRAG storage directory."""

import shutil
from pathlib import Path

from pipeline.config import DB_PATH, LIGHTRAG_STORAGE_DIR


def hard_reset() -> dict:
    """Delete all persistent state so the system can be re-initialized from scratch.

    Removes:
    - The SQLite database file and any WAL/SHM sidecar files
    - The entire LightRAG storage directory

    Returns a dict with the list of paths that were actually deleted.
    """
    deleted: list[str] = []

    # SQLite main file + WAL sidecar files
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(DB_PATH) + suffix)
        if p.exists():
            p.unlink()
            deleted.append(str(p))

    # LightRAG knowledge graph storage
    if LIGHTRAG_STORAGE_DIR.exists():
        shutil.rmtree(LIGHTRAG_STORAGE_DIR)
        deleted.append(str(LIGHTRAG_STORAGE_DIR))

    return {"deleted": deleted}
