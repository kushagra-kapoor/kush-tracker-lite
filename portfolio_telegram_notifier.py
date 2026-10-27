"""
Portfolio Telegram Notifier Module
==================================
Handles 100% local Telegram messaging for Kush Tracker's daily portfolio briefings.
Uses Python's standard library (urllib.request) to eliminate external dependencies.
Features smart message chunking, Day 0 directory initialization, and idempotent
once-per-day guardrail tracking.
"""

import os
import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, date
from pathlib import Path
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SENT_LOG_FILE = DATA_DIR / "telegram_sent_log.json"

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_LENGTH = 3950  # Safe margin below Telegram's 4096 limit


def _load_env_if_needed():
    """Ensure .env is loaded if dotenv is available."""
    try:
        from dotenv import load_dotenv
        env_path = BASE_DIR / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
    except ImportError:
        pass


_load_env_if_needed()


def get_telegram_credentials() -> Tuple[str, str, bool]:
    """Retrieves token, chat_id, and auto-alert toggle from environment."""
    _load_env_if_needed()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    enabled_str = os.environ.get("ENABLE_TELEGRAM_DAILY_ALERT", "true").strip().lower()
    is_enabled = enabled_str in ("true", "1", "yes")
    return token, chat_id, is_enabled


def is_telegram_configured() -> bool:
    """Returns True if bot token and chat_id are present."""
    token, chat_id, _ = get_telegram_credentials()
    return bool(token and chat_id)


def ensure_data_dir():
    """Day 0 Self-Healing: Creates data/ dir and default sent log if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SENT_LOG_FILE.exists():
        try:
            with open(SENT_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump({"last_sent_date": None, "history": []}, f, indent=2)
        except Exception as e:
            logger.warning(f"[Telegram Notifier] Failed to initialize sent log: {e}")


def has_sent_today(target_date: Optional[str] = None) -> bool:
    """Checks if a daily digest was already dispatched on target_date (YYYY-MM-DD)."""
    ensure_data_dir()
    if target_date is None:
        target_date = date.today().isoformat()
    if not SENT_LOG_FILE.exists():
        return False
    try:
        with open(SENT_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("last_sent_date") == target_date
    except Exception:
        return False


def mark_sent_today(target_date: Optional[str] = None):
    """Records that a digest was dispatched today to prevent duplicate spam on reruns."""
    ensure_data_dir()
    if target_date is None:
        target_date = date.today().isoformat()
    now_ts = datetime.now().isoformat()
    
    current_data = {"last_sent_date": None, "history": []}
    if SENT_LOG_FILE.exists():
        try:
            with open(SENT_LOG_FILE, "r", encoding="utf-8") as f:
                current_data = json.load(f)
        except Exception:
            current_data = {"last_sent_date": None, "history": []}
            
    current_data["last_sent_date"] = target_date
    history = current_data.get("history", [])
    history.append({"date": target_date, "timestamp": now_ts})
    # Keep last 30 entries
    current_data["history"] = history[-30:]
    
    try:
        with open(SENT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2)
    except Exception as e:
        logger.warning(f"[Telegram Notifier] Failed to update sent log: {e}")


def _chunk_message(text: str) -> List[str]:
    """Splits a large HTML message into safe chunks <= MAX_MESSAGE_LENGTH (4000 chars)."""
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]

    chunks = []
    current_chunk = []
    current_len = 0

    lines = text.split("\n")
    for line in lines:
        line_len = len(line) + 1  # include \n
        if current_len + line_len > MAX_MESSAGE_LENGTH:
            if current_chunk:
                chunks.append("\n".join(current_chunk).strip())
                current_chunk = [line]
                current_len = line_len
            else:
                # Fallback for an exceptionally long single line
                chunks.append(line[:MAX_MESSAGE_LENGTH])
                current_chunk = [line[MAX_MESSAGE_LENGTH:]]
                current_len = len(current_chunk[0]) + 1
        else:
            current_chunk.append(line)
            current_len += line_len

    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def send_telegram_portfolio_message(
    message_text: str,
    token: Optional[str] = None,
    chat_id: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Dispatches a formatted HTML message to Telegram.
    Handles multi-part chunking, timeouts, and error diagnostics.
    
    Returns:
        (success: bool, error_message: str)
    """
    if not token or not chat_id:
        cfg_token, cfg_chat, _ = get_telegram_credentials()
        token = token or cfg_token
        chat_id = chat_id or cfg_chat

    if not token or not chat_id:
        return False, "Telegram Bot Token or Chat ID not configured in .env."

    chunks = _chunk_message(message_text)
    total_parts = len(chunks)
    url = TELEGRAM_API_URL.format(token=token)

    for idx, chunk in enumerate(chunks, 1):
        # Add page count header if message had to be split
        payload_text = chunk
        if total_parts > 1:
            payload_text = f"<i>[Part {idx}/{total_parts}]</i>\n{chunk}"

        payload = {
            "chat_id": chat_id,
            "text": payload_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if not result.get("ok"):
                    err = f"Telegram API error: {result.get('description')}"
                    logger.error(err)
                    return False, err
        except urllib.error.HTTPError as e:
            try:
                err_body = json.loads(e.read().decode("utf-8"))
                desc = err_body.get("description", str(e))
            except Exception:
                desc = str(e)
            err_msg = f"HTTP {e.code}: {desc}"
            logger.error(f"[Telegram Notifier] {err_msg}")
            return False, err_msg
        except Exception as e:
            err_msg = f"Network exception: {e}"
            logger.error(f"[Telegram Notifier] {err_msg}")
            return False, err_msg

    return True, f"Successfully sent {total_parts} part(s) to Telegram."
