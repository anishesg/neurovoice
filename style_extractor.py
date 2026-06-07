"""Style Extractor — Learn communication style from iMessages and Gmail.

Reads the local iMessage database, extracts conversation pairs,
builds a style profile, and selects few-shot examples for LLM prompting.
"""

import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from config import IMESSAGE_DB

MAC_EPOCH = datetime(2001, 1, 1)


@dataclass
class ConversationExample:
    incoming: str
    response: str
    contact: str = ""
    context: str = "casual"
    timestamp: datetime = None


@dataclass
class StyleProfile:
    avg_response_length: float = 0.0
    uses_lowercase: bool = False
    uses_exclamations: bool = False
    uses_emojis: bool = False
    emoji_frequency: float = 0.0
    avg_words_per_message: float = 0.0
    punctuation_rate: float = 0.0
    common_greetings: list = field(default_factory=list)
    common_signoffs: list = field(default_factory=list)
    formality_score: float = 0.5
    sample_count: int = 0

    def to_prompt_description(self):
        parts = []
        if self.uses_lowercase:
            parts.append("tends to write in lowercase")
        if self.uses_exclamations:
            parts.append("uses exclamation marks frequently")
        if self.uses_emojis:
            parts.append("uses emojis naturally in conversation")
        if self.avg_words_per_message < 10:
            parts.append("keeps messages very brief (under 10 words)")
        elif self.avg_words_per_message < 25:
            parts.append("writes moderately concise messages")
        else:
            parts.append("writes detailed, longer messages")
        if self.formality_score < 0.3:
            parts.append("very casual and informal tone")
        elif self.formality_score > 0.7:
            parts.append("formal and professional tone")
        if self.common_greetings:
            parts.append(f"typical greetings: {', '.join(self.common_greetings[:3])}")
        return "; ".join(parts) if parts else "natural conversational style"


def _convert_date(mac_ts):
    if not mac_ts or mac_ts == 0:
        return None
    return MAC_EPOCH + timedelta(seconds=mac_ts / 1_000_000_000)


def _extract_attributed_body(blob):
    if not blob:
        return None
    try:
        text = blob.split(b"NSString")[1]
        text = text[5:]
        if text[0:1] == b"\x81":
            length = int.from_bytes(text[1:3], "little")
            text = text[3:3 + length]
        else:
            length = text[0]
            text = text[1:1 + length]
        return text.decode("utf-8", errors="replace")
    except (IndexError, ValueError):
        return None


def _get_text(row):
    if row["text"]:
        return row["text"]
    return _extract_attributed_body(row["attributedBody"])


def _has_emoji(text):
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0\U0001FA00-\U0001FA6F]+",
        flags=re.UNICODE,
    )
    return bool(emoji_pattern.search(text))


def extract_imessage_conversations(limit=5000, min_length=3):
    """Read iMessage database and extract conversation pairs."""
    db_path = os.path.expanduser(IMESSAGE_DB)
    if not Path(db_path).exists():
        print(f"[Style] iMessage DB not found at {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
    SELECT
        m.text,
        m.attributedBody,
        m.is_from_me,
        m.date,
        m.service,
        m.associated_message_type,
        h.id AS contact_id,
        c.chat_identifier,
        c.display_name AS chat_name,
        c.style AS chat_style
    FROM message m
    LEFT JOIN handle h ON m.handle_id = h.ROWID
    LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    LEFT JOIN chat c ON cmj.chat_id = c.ROWID
    WHERE m.associated_message_type = 0
    ORDER BY m.date ASC
    LIMIT ?
    """
    rows = conn.execute(query, (limit,)).fetchall()
    conn.close()

    pairs = []
    prev = None
    for row in rows:
        text = _get_text(row)
        if not text or len(text.strip()) < min_length:
            prev = None
            continue

        if row["is_from_me"] and prev and not prev["is_from_me"]:
            prev_text = _get_text(prev)
            if prev_text and len(prev_text.strip()) >= min_length:
                contact = prev.get("contact_id") or prev.get("chat_identifier") or "unknown"
                pairs.append(ConversationExample(
                    incoming=prev_text.strip(),
                    response=text.strip(),
                    contact=str(contact),
                    context="casual",
                    timestamp=_convert_date(row["date"]),
                ))
        prev = row

    return pairs


def build_style_profile(pairs):
    """Analyze conversation pairs to build a style profile."""
    if not pairs:
        return StyleProfile()

    responses = [p.response for p in pairs]
    lengths = [len(r) for r in responses]
    word_counts = [len(r.split()) for r in responses]

    lowercase_count = sum(1 for r in responses if r[0].islower())
    excl_count = sum(1 for r in responses if "!" in r)
    emoji_count = sum(1 for r in responses if _has_emoji(r))
    punct_count = sum(1 for r in responses if r.rstrip()[-1:] in ".!?")

    greetings = {}
    for r in responses:
        first_word = r.split()[0].lower() if r.split() else ""
        if first_word in ("hey", "hi", "hello", "yo", "sup", "haha", "lol",
                         "yeah", "yea", "nah", "heyy", "hii"):
            greetings[first_word] = greetings.get(first_word, 0) + 1

    formal_markers = sum(1 for r in responses
                        if any(m in r.lower() for m in
                              ("dear ", "regards", "sincerely", "best,")))
    casual_markers = sum(1 for r in responses
                        if any(m in r.lower() for m in
                              ("lol", "haha", "omg", "tbh", "ngl", "fr")))

    n = len(responses)
    formality = 0.5
    if formal_markers + casual_markers > 0:
        formality = formal_markers / (formal_markers + casual_markers)

    top_greetings = sorted(greetings, key=greetings.get, reverse=True)[:5]

    return StyleProfile(
        avg_response_length=sum(lengths) / n,
        uses_lowercase=lowercase_count / n > 0.5,
        uses_exclamations=excl_count / n > 0.3,
        uses_emojis=emoji_count / n > 0.15,
        emoji_frequency=emoji_count / n,
        avg_words_per_message=sum(word_counts) / n,
        punctuation_rate=punct_count / n,
        common_greetings=top_greetings,
        formality_score=formality,
        sample_count=n,
    )


def select_few_shot_examples(pairs, n=8, context="casual"):
    """Select diverse, representative examples for few-shot prompting."""
    filtered = [p for p in pairs if p.context == context]
    if not filtered:
        filtered = pairs

    if len(filtered) <= n:
        return filtered

    lengths = [len(p.response) for p in filtered]
    median_len = sorted(lengths)[len(lengths) // 2]

    scored = []
    for p in filtered:
        score = 0
        resp_len = len(p.response)
        score -= abs(resp_len - median_len) / (median_len + 1)
        if 5 < len(p.incoming) < 200:
            score += 1
        if 5 < resp_len < 300:
            score += 1
        scored.append((score, p))

    scored.sort(key=lambda x: -x[0])
    step = max(1, len(scored) // n)
    selected = [scored[i * step][1] for i in range(min(n, len(scored)))]
    return selected


def format_examples_for_prompt(examples):
    """Format conversation examples as a string for LLM system prompt."""
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  They said: \"{ex.incoming}\"")
        lines.append(f"  You replied: \"{ex.response}\"")
    return "\n".join(lines)


def extract_imessage_style():
    """One-call convenience: extract style profile + examples from iMessages."""
    print("[Style] Reading iMessage database...")
    pairs = extract_imessage_conversations(limit=10000)
    print(f"[Style] Found {len(pairs)} conversation pairs")

    profile = build_style_profile(pairs)
    examples = select_few_shot_examples(pairs, n=8)

    return {
        "profile": profile,
        "examples": examples,
        "pairs_count": len(pairs),
    }
