"""At-rest encryption for conversation memory.

Uses Fernet (AES-128-CBC + HMAC). The key comes from TUNDA_MEMORY_KEY,
TUNDA_MEMORY_PASSPHRASE, or a generated file that is never committed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64

    _CRYPTO = True
except ImportError:
    Fernet = None  # type: ignore
    InvalidToken = Exception  # type: ignore
    _CRYPTO = False

ENVELOPE_VERSION = 1
KEY_ENV = "TUNDA_MEMORY_KEY"
PASSPHRASE_ENV = "TUNDA_MEMORY_PASSPHRASE"
_SALT = b"tunda-memory-v1"


def crypto_available() -> bool:
    return _CRYPTO


def derive_key_from_passphrase(passphrase: str) -> bytes:
    if not _CRYPTO:
        raise RuntimeError("cryptography is not installed")
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=_SALT, iterations=480_000)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def load_or_create_key(key_file: str | Path) -> bytes:
    """Return a Fernet key from env, passphrase, or a local key file."""
    if not _CRYPTO:
        raise RuntimeError("Install cryptography to encrypt memory at rest.")

    env_key = os.environ.get(KEY_ENV, "").strip()
    if env_key:
        return env_key.encode("utf-8") if len(env_key) != 44 else env_key.encode("ascii")

    passphrase = os.environ.get(PASSPHRASE_ENV, "").strip()
    if passphrase:
        return derive_key_from_passphrase(passphrase)

    path = Path(key_file)
    if path.exists():
        return path.read_bytes().strip()

    key = Fernet.generate_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    logger.info("Created memory encryption key at %s", path)
    return key


class EncryptedJsonStore:
    """Read/write JSON, optionally Fernet-encrypted on disk."""

    def __init__(self, path: str | Path, encrypt: bool = True, key: Optional[bytes] = None, key_file: str = "data/.memory_key"):
        self.path = Path(path)
        self.encrypt = encrypt
        self._fernet = None
        if encrypt:
            if not _CRYPTO:
                raise RuntimeError("Memory encryption is on, but cryptography is not installed.")
            material = key or load_or_create_key(key_file)
            self._fernet = Fernet(material)

    def read(self) -> Any:
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("tunda_memory") == ENVELOPE_VERSION:
            if not self._fernet:
                raise RuntimeError("Encrypted memory file found, but encryption is disabled.")
            try:
                plain = self._fernet.decrypt(data["payload"].encode("ascii"))
            except InvalidToken as exc:
                raise RuntimeError("Memory key does not match encrypted conversation file.") from exc
            return json.loads(plain.decode("utf-8"))
        return data

    def write(self, obj: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.encrypt and self._fernet:
            token = self._fernet.encrypt(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
            envelope = {
                "tunda_memory": ENVELOPE_VERSION,
                "alg": "fernet",
                "payload": token.decode("ascii"),
            }
            self.path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
        else:
            self.path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
