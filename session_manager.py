"""
Session Manager for Kush Tracker Lite.
Provides high-security, single-day session persistence across mobile refreshes,
tab suspensions, and app switches.

Security Model:
1. High-Entropy 256-bit Cryptographic Tokens (CSPRNG via secrets module).
2. One-Way SHA-256 Hashing: Only token hashes are stored in the database.
3. Single-Day Expiry: Sessions strictly expire at midnight (end of day).
4. Instant Invalidation: Sign Out purges hash from DB immediately.
"""
import secrets
import hashlib
from datetime import datetime, time
from typing import Tuple

from database import get_connection, _fetch_one_dict, safe_execute


def _ensure_session_table():
    """Ensure app_sessions table and index exist in database on any connection."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        safe_execute(cursor, '''
            CREATE TABLE IF NOT EXISTS app_sessions (
                token_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                device_fingerprint TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_active TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        safe_execute(cursor, '''
            CREATE INDEX IF NOT EXISTS idx_sessions_hash ON app_sessions(token_hash)
        ''')
        conn.commit()
    except Exception as e:
        print(f"[SessionManager] Table init warning: {e}")
    finally:
        conn.close()


def _hash_token(token: str) -> str:
    """Return SHA-256 hexadecimal digest of raw session token."""
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def get_end_of_day_iso() -> str:
    """Return ISO-8601 string for 23:59:59 of current calendar day."""
    today = datetime.now().date()
    end_of_day = datetime.combine(today, time(23, 59, 59))
    return end_of_day.isoformat()


def create_secure_session(username: str) -> str:
    """
    Generate high-entropy 256-bit token.
    Stores only the SHA-256 hash in database.
    Returns the raw token to be stored in the browser's URL query params.
    """
    _ensure_session_table()
    raw_token = f"kt_{secrets.token_urlsafe(32)}"
    token_hash = _hash_token(raw_token)
    expires_at = get_end_of_day_iso()
    now_str = datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO app_sessions 
               (token_hash, username, expires_at, device_fingerprint, created_at, last_active) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (token_hash, username, expires_at, "web_client", now_str, now_str)
        )
        conn.commit()
        return raw_token
    except Exception as e:
        print(f"[Security] Failed to create secure session: {e}")
        return ""
    finally:
        conn.close()


def validate_secure_session(raw_token: str) -> Tuple[bool, str]:
    """
    Validate session:
    1. Checks if SHA-256 hash exists in app_sessions.
    2. Ensures current time <= expires_at (valid today).
    3. Updates last_active timestamp on success.
    """
    if not raw_token or not isinstance(raw_token, str):
        return False, ""

    _ensure_session_table()
    token_hash = _hash_token(raw_token.strip())

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT username, expires_at FROM app_sessions WHERE token_hash = ?",
            (token_hash,)
        )
        row = _fetch_one_dict(cursor)
        if not row:
            return False, ""

        # Single-day expiration check
        try:
            expires_at = datetime.fromisoformat(row["expires_at"])
            if datetime.now() > expires_at:
                # Expired for the day: clean up
                cursor.execute("DELETE FROM app_sessions WHERE token_hash = ?", (token_hash,))
                conn.commit()
                return False, ""
        except Exception:
            cursor.execute("DELETE FROM app_sessions WHERE token_hash = ?", (token_hash,))
            conn.commit()
            return False, ""

        # Valid: Touch last_active
        now_str = datetime.now().isoformat()
        cursor.execute("UPDATE app_sessions SET last_active = ? WHERE token_hash = ?", (now_str, token_hash))
        conn.commit()
        return True, row.get("username", "admin")
    except Exception as e:
        print(f"[Security] Validation error: {e}")
        return False, ""
    finally:
        conn.close()


def revoke_secure_session(raw_token: str) -> bool:
    """Revoke session token by deleting its hash from the database."""
    if not raw_token or not isinstance(raw_token, str):
        return False
    _ensure_session_table()
    token_hash = _hash_token(raw_token.strip())
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM app_sessions WHERE token_hash = ?", (token_hash,))
        conn.commit()
        return True
    except Exception as e:
        print(f"[Security] Error revoking session: {e}")
        return False
    finally:
        conn.close()


def cleanup_expired_sessions() -> int:
    """Purge sessions that have passed their single-day expiry."""
    _ensure_session_table()
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now_str = datetime.now().isoformat()
        cursor.execute("DELETE FROM app_sessions WHERE expires_at < ?", (now_str,))
        conn.commit()
        return cursor.rowcount if hasattr(cursor, "rowcount") else 0
    except Exception:
        return 0
    finally:
        conn.close()
