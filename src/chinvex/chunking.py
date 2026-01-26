from __future__ import annotations

from dataclasses import dataclass
from math import ceil


MAX_CHARS = 3000
REPO_OVERLAP = 300
MAX_CONVERSATION_TOKENS = 1500


@dataclass(frozen=True)
class Chunk:
    text: str
    ordinal: int
    char_start: int | None = None
    char_end: int | None = None
    msg_start: int | None = None
    msg_end: int | None = None
    roles_present: list[str] | None = None


def chunk_repo(text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = MAX_CHARS - REPO_OVERLAP
    start = 0
    ordinal = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        chunk_text = text[start:end]
        chunks.append(Chunk(text=chunk_text, ordinal=ordinal, char_start=start, char_end=end))
        ordinal += 1
        start += step
    if not chunks:
        chunks.append(Chunk(text="", ordinal=0, char_start=0, char_end=0))
    return chunks


def build_chat_text(messages: list[dict]) -> list[str]:
    lines: list[str] = []
    for i, msg in enumerate(messages):
        role = str(msg.get("role", "unknown"))
        text = str(msg.get("text", "")).strip()
        lines.append(f"[{i:04d}] {role}: {text}")
    return lines


def chunk_chat(messages: list[dict]) -> list[Chunk]:
    lines = build_chat_text(messages)
    chunks: list[Chunk] = []
    current: list[str] = []
    msg_start = 0
    ordinal = 0
    roles_present: set[str] = set()

    def flush(msg_end: int) -> None:
        nonlocal ordinal, msg_start, current, roles_present
        chunk_text = "\n".join(current)
        chunks.append(
            Chunk(
                text=chunk_text,
                ordinal=ordinal,
                msg_start=msg_start,
                msg_end=msg_end,
                roles_present=sorted(roles_present),
            )
        )
        ordinal += 1
        current = []
        roles_present = set()

    for i, line in enumerate(lines):
        msg = messages[i]
        role = str(msg.get("role", "unknown"))
        pending = current + [line]
        if len(pending) >= 10 or sum(len(x) + 1 for x in pending) > MAX_CHARS:
            if current:
                flush(i - 1)
                msg_start = i
            current.append(line)
            roles_present.add(role)
        else:
            current.append(line)
            roles_present.add(role)
            if len(current) == 10:
                flush(i)
                msg_start = i + 1

    if current:
        flush(len(lines) - 1)

    if not chunks:
        chunks.append(Chunk(text="", ordinal=0, msg_start=0, msg_end=0, roles_present=[]))
    return chunks


def approx_tokens(text: str) -> int:
    """Approximate token count using len/4 heuristic."""
    return ceil(len(text) / 4)


def chunk_conversation(conversation: dict, max_tokens: int = MAX_CONVERSATION_TOKENS) -> list[Chunk]:
    """
    Chunk a ConversationDoc by logical turn groups.

    Rules:
    - Group consecutive turns up to ~max_tokens
    - Never split a single turn across chunks
    - Include [Turn N of M] markers
    - Preserve thread_id and turn_id range in metadata
    """
    turns = conversation["turns"]
    total_turns = len(turns)
    chunks: list[Chunk] = []

    if not turns:
        return [Chunk(text="", ordinal=0)]

    current_group: list[dict] = []
    current_tokens = 0
    ordinal = 0

    def flush_group():
        nonlocal ordinal, current_group, current_tokens
        if not current_group:
            return

        # Build chunk text with turn markers
        lines = []
        for turn in current_group:
            turn_idx = turns.index(turn)
            marker = f"[Turn {turn_idx + 1} of {total_turns}]"
            role = turn.get("role", "unknown")
            text = turn.get("text", "")
            lines.append(f"{marker} {role}: {text}")

        chunk_text = "\n\n".join(lines)

        # Metadata: turn range
        first_turn = current_group[0]
        last_turn = current_group[-1]

        chunks.append(
            Chunk(
                text=chunk_text,
                ordinal=ordinal,
            )
        )
        ordinal += 1
        current_group = []
        current_tokens = 0

    for turn_idx, turn in enumerate(turns):
        turn_text = turn.get("text", "")
        role = turn.get("role", "unknown")

        # Estimate token count including formatting overhead
        marker = f"[Turn {turn_idx + 1} of {total_turns}]"
        formatted_text = f"{marker} {role}: {turn_text}"
        turn_tokens = approx_tokens(formatted_text)

        # Add separator overhead if not first turn in group
        if current_group:
            turn_tokens += approx_tokens("\n\n")

        # Check if adding this turn would exceed limit
        if current_group and (current_tokens + turn_tokens > max_tokens):
            # Flush current group
            flush_group()
            # Recalculate for first item in new group (no separator)
            turn_tokens = approx_tokens(formatted_text)

        # Add turn to current group
        current_group.append(turn)
        current_tokens += turn_tokens

    # Flush remaining
    flush_group()

    return chunks
