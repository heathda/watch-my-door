#!/usr/bin/env python3
"""
Email notifications, with optional image attachment.

Gmail over SMTP_SSL. For Gmail you must use a 16-char App Password
(Google Account -> Security -> App passwords), not your normal login password.
"""

import logging
import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

logger = logging.getLogger(__name__)

# Credential Manager "Internet or network address" / generic-credential name.
KEYRING_SERVICE = os.getenv("KEYRING_SERVICE", "watch-my-door")


def resolve_password(gmail_user: str | None) -> str | None:
    """
    Get the Gmail App Password from Windows Credential Manager (via keyring),
    falling back to the GMAIL_PASSWORD env var.

    Store it once (as the account BlueIris runs under) with:
        keyring set watch-my-door <gmail_user>
    Then GMAIL_PASSWORD can be removed from .env entirely.
    """
    if gmail_user:
        try:
            import keyring

            pw = keyring.get_password(KEYRING_SERVICE, gmail_user)
            if pw:
                return pw
        except Exception as e:  # keyring missing, no backend, locked vault, etc.
            logger.warning(f"Credential Manager lookup failed ({e}); using .env")
    return os.getenv("GMAIL_PASSWORD")


def _attach_image(msg: EmailMessage, image_path: Path) -> None:
    """Attach the alert image, guessing its MIME type from the extension."""
    ctype, _ = mimetypes.guess_type(image_path.name)
    maintype, subtype = (ctype or "image/jpeg").split("/", 1)
    msg.add_attachment(
        image_path.read_bytes(),
        maintype=maintype,
        subtype=subtype,
        filename=image_path.name,
    )


def send_email(subject: str, body: str, image_path: str | None = None) -> bool:
    """
    Send a notification email. Returns True on success, False on any failure
    (and logs it) -- callers should never let a notification problem crash the
    pipeline.
    """
    # Read at call time, not import time, so .env is already loaded.
    gmail_user = os.getenv("GMAIL_USER")
    gmail_password = resolve_password(gmail_user)
    # NOTIFY_EMAIL may hold one address or a comma/semicolon-separated list.
    notify_email = os.getenv("NOTIFY_EMAIL") or ""
    recipients = [addr.strip() for addr in notify_email.replace(";", ",").split(",") if addr.strip()]

    if not all([gmail_user, gmail_password, recipients]):
        logger.error(
            "Email credentials not configured. Set GMAIL_USER/NOTIFY_EMAIL in "
            ".env and store the password in Windows Credential Manager "
            "(keyring) or GMAIL_PASSWORD in .env."
        )
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    if image_path:
        path = Path(image_path)
        if path.is_file():
            try:
                _attach_image(msg, path)
            except OSError as e:
                logger.warning(f"Could not attach image {path}: {e}")
        else:
            logger.warning(f"Image path does not exist, sending text only: {path}")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(gmail_user, gmail_password)
            server.send_message(msg)
        logger.info(f"Email notification sent to {', '.join(recipients)}")
        return True
    except smtplib.SMTPException as e:
        logger.error(f"Failed to send email: {e}")
        return False
