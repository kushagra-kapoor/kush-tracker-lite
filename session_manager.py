"""
Session Manager for Kush Tracker Lite.
Provides high-security, single-day session persistence across mobile refreshes,
tab suspensions, and app switches.

Security Model:
1. One-Way SHA-256 Hashing: Only token hashes are stored in the database.
2. Single-Day Expiry: Sessions strictly expire at midnight (end of day).
3. Client Fingerprinting: Binds token to client User-Agent to prevent link theft.
4. Instant Invalidation: Sign Out purges hash from DB immediately.
"""
import secrets
import hashlib
from datetime import datetime, time
from typing import Tuple

from database import get_connection, _fetch_one_dict


def _hash_token(token: str) -> str:
    """Return SHA-256 hexadecimal digest of raw session token."""
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def _get_client_fingerprint() -> str:
    """Generate a lightweight client fingerprint from User-Agent header."""
    try:
        import streamlit as st
        if hasattr(st, "context") and hasattr(st.context, "headers"):
            ua = st.context.headers.get("user-agent", "")
            if ua:
                return hashlib.sha256(ua.encode('utf-8')).hexdigest()[:32]
    except Exception:
        pass
    return "device_generic"


def get_end_of_day_iso() -> str:
    """Return ISO-8601 string for 23:59:59 of current calendar day."""
    today = datetime.now().date()
    end_of_day = datetime.combine(today, time(23, 59, 59))
    return end_of_day.isoformat()


def create_secure_session(username: str) -> str:
    """
    Generate high-entropy 256-bit token.
    Stores only the SHA-256 hash and client fingerprint in database.
    Returns the raw token to be stored in the browser's URL query params.
    """
    raw_token = f"kt_{secrets.token_urlsafe(32)}"
    token_hash = _hash_token(raw_token)
    fingerprint = _get_client_fingerprint()
    expires_at = get_end_of_day_iso()
    now_str = datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO app_sessions 
               (token_hash, username, expires_at, device_fingerprint, created_at, last_active) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (token_hash, username, expires_at, fingerprint, now_str, now_str)
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
    3. Verifies client fingerprint matches (anti-hijack protection).
    4. Updates last_active timestamp on success.
    """
    if not raw_token or not isinstance(raw_token, str):
        return False, ""

    token_hash = _hash_token(raw_token.strip())
    current_fingerprint = _get_client_fingerprint()

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT username, expires_at, device_fingerprint FROM app_sessions WHERE token_hash = ?",
            (token_hash,)
        )
        row = _fetch_one_dict(cursor)
        if not row:
            return False, ""

        # 1. Single-day expiration check
        try:
            expires_at = datetime.fromisoformat(row["expires_at"])
            if datetime.now() > expires_at:
                # Expired for the day: clean up
                cursor.execute("DELETE FROM app_sessions WHERE token_hash = ?", (token_hash,))
                conn.commit()
                return False, ""
        except Exception:
            # Corrupted date string
            cursor.execute("DELETE FROM app_sessions WHERE token_hash = ?", (token_hash,))
            conn.commit()
            return False, ""

        # 2. Client Device Fingerprint Check (Anti-Tamper / Anti-Theft)
        stored_fingerprint = row.get("device_fingerprint", "")
        if (
            stored_fingerprint
            and stored_fingerprint != "device_generic"
            and current_fingerprint != "device_generic"
        ):
            if stored_fingerprint != current_fingerprint:
                print("[Security Warning] Session rejected: Client fingerprint mismatch.")
                cursor.execute("DELETE FROM app_sessions WHERE token_hash = ?", (token_hash,))
                conn.commit()
                return False, ""

        # 3. Touch last_active
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
