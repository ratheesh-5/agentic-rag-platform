"""Upload validation: never trust a file because of its name.

Every uploaded file is checked before it touches the disk:
- file name: only the base name is kept (no "../" path tricks), unusual characters are replaced,
  and the extension must be one we can index (.md .txt .pdf .docx);
- size: rejected above the configured limit (the API reads at most limit + 1 bytes);
- content: the bytes must match the claimed type ("magic bytes" / file signature):
  a PDF starts with "%PDF-", a DOCX is a ZIP containing word/document.xml, and MD/TXT must be
  valid UTF-8 text without NUL bytes. A renamed .exe is therefore refused even if called x.pdf.

Plain functions raising UploadError, so they are unit-testable without a web server. The API
maps UploadError.status to the HTTP status code (413 too large, 415 type, 422 content).

Prompt-injection scan (security plan B6, layer 2): `suspicious_lines` finds text written to
give orders to an AI ("ignore all previous instructions", "you are now ...", "reveal your system
prompt", chat-template tokens). Layer 1 is the prompt rule UNTRUSTED_PASSAGES (passages are
data, never instructions). The scan only WARNS - the document is indexed and the admin decides:
real documents may quote such phrases (a security handbook explaining attacks). The patterns
need an order-like shape, so text merely ABOUT prompt injection (like the sample runbook) is
not flagged. Pattern matching, no AI call: fast and free, but a determined attacker can reword
an attack - which is why the prompt rule stays the main defence.
"""

import io
import re
import zipfile

ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

# Order-like phrases aimed at an AI model (case-insensitive). Checked against the sample corpus:
# no false alarms there, including the runbook's section about prompt injection.
_INJECTION = re.compile(
    "|".join(
        f"(?:{p})"
        for p in [
            # "ignore all previous instructions", "disregard the rules above", "forget your rules"
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,25}\b(previous|prior|above|earlier"
            r"|preceding|all|any|your|the)\s+(instructions|rules|prompts?|directions|guidelines)\b",
            r"\byou\s+are\s+now\s+(a|an|in|the)\b",  # "you are now an unrestricted assistant"
            r"\b(reveal|print|show|repeat|output|leak)\b[^.\n]{0,20}\b(system\s+prompt"
            r"|hidden\s+(prompt|instructions)|your\s+instructions)\b",
            r"\b(do\s+not|don'?t|never)\s+(tell|inform)\s+the\s+user\b",  # hide from the user
            r"\bnew\s+(system\s+)?instructions\s*:",
            r"<\|im_start\|>|\[/?INST\]|<<SYS>>|</?system>",  # chat-template control tokens
        ]
    ),
    re.IGNORECASE,
)
MAX_WARNINGS = 3  # lines shown per document; enough to find the spot


def suspicious_lines(text: str) -> list[str]:
    """Lines that look like instructions to an AI (B6), shortened for display; [] = clean."""
    found = [line.strip() for line in text.splitlines() if _INJECTION.search(line)]
    return [line if len(line) <= 160 else line[:157] + "..." for line in found[:MAX_WARNINGS]]


class UploadError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def safe_filename(name: str) -> str:
    """Base name only, safe characters only, allowed extension required."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]  # drop any directory part (Windows or Unix)
    base = _UNSAFE.sub("_", base).strip("._")[:100]
    if "." not in base:
        raise UploadError(415, "File needs one of these extensions: .md .txt .pdf .docx")
    ext = "." + base.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(415, f"Unsupported file type {ext}; allowed: .md .txt .pdf .docx")
    return base.rsplit(".", 1)[0] + ext


def check_content(filename: str, data: bytes, max_bytes: int) -> None:
    """Size and file-signature checks for an already-sanitized file name."""
    if not data:
        raise UploadError(422, "The file is empty")
    if len(data) > max_bytes:
        raise UploadError(413, f"File too large; the limit is {max_bytes // (1024 * 1024)} MB")
    ext = "." + filename.rsplit(".", 1)[1]
    if ext == ".pdf" and not data.startswith(b"%PDF-"):
        raise UploadError(422, "Not a valid PDF (missing %PDF- header)")
    if ext == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:  # read the ZIP from memory
                if "word/document.xml" not in z.namelist():
                    raise UploadError(422, "Not a valid DOCX (no word/document.xml)")
        except zipfile.BadZipFile:
            raise UploadError(422, "Not a valid DOCX (not a ZIP container)") from None
    if ext in (".md", ".txt"):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise UploadError(422, "Text files must be UTF-8") from None
        if "\x00" in text:
            raise UploadError(422, "Text file contains binary data")
