from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from math import ceil


def chunk_key(text: str) -> str:
    """
    Generate stable key for chunk embedding lookup.

    Normalizes whitespace before hashing to handle minor formatting differences.

    Returns:
        16-character hex string (sha256 prefix)
    """
    # Collapse all whitespace to single spaces
    normalized = ' '.join(text.split())
    # Hash normalized text
    hash_bytes = hashlib.sha256(normalized.encode('utf-8')).digest()
    # Return first 16 hex chars (64 bits)
    return hash_bytes.hex()[:16]


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


def chunk_with_overlap(text: str, size: int = 3000, overlap: int = 300) -> list[tuple[int, int]]:
    """
    Return list of (start, end) positions for chunks with overlap.

    Generic fallback for prose and unknown file types.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append((start, end))
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_generic_file(text: str, size: int = 3000, overlap: int = 300) -> list[Chunk]:
    """
    Chunk generic text file with overlap.

    Used for: txt, unknown extensions, prose files.
    """
    positions = chunk_with_overlap(text, size, overlap)
    chunks = []
    for ordinal, (start, end) in enumerate(positions):
        chunk_text = text[start:end]
        chunks.append(Chunk(
            text=chunk_text,
            ordinal=ordinal,
            char_start=start,
            char_end=end,
        ))
    return chunks


# Semantic boundary priorities (pattern, score)
SPLIT_PRIORITIES = [
    (r'\n## ', 100),           # Markdown H2
    (r'\n### ', 90),           # Markdown H3
    (r'\n---\n', 85),          # Markdown horizontal rule
    (r'\nclass ', 80),         # Python class (heuristic)
    (r'\ndef ', 75),           # Python function (heuristic)
    (r'\nasync def ', 75),     # Python async function
    (r'\nfunction ', 75),      # JS function (heuristic)
    (r'\nasync function ', 75),# JS async function
    (r'\nconst \w+ = \(', 72), # JS arrow function
    (r'\nconst \w+ = async \(', 72),  # JS async arrow
    (r'\nconst \w+ = ', 70),   # JS const declaration
    (r'\nexport default ', 70),# JS/TS default export
    (r'\nexport ', 68),        # JS/TS named export
    (r'\nmodule\.exports', 68),# CommonJS export
    (r'\n\n\n', 60),           # Multiple blank lines
    (r'\n\n', 50),             # Paragraph break
    (r'\n', 10),               # Line break (last resort)
]


def find_best_split(text: str, target_pos: int, size: int = 3000) -> int:
    """
    Find best split point near target_pos.

    Searches within ±window chars for highest-priority boundary.
    Always returns a position at a newline or other boundary.
    """
    window = int(size * 0.5)  # 50% window for better boundary detection
    search_start = max(0, target_pos - window)
    search_end = min(len(text), target_pos + window)
    search_region = text[search_start:search_end]

    best_score = -1
    best_pos = None

    for pattern, score in SPLIT_PRIORITIES:
        for match in re.finditer(pattern, search_region):
            pos = search_start + match.start()
            # Prefer splits closer to target
            distance_penalty = abs(pos - target_pos) / window * 10
            effective_score = score - distance_penalty
            if effective_score > best_score:
                best_score = effective_score
                best_pos = pos

    # If no boundary found, fallback to target position
    if best_pos is None:
        best_pos = target_pos

    return best_pos


def chunk_markdown_file(text: str, size: int = 3000, overlap: int = 300) -> list[Chunk]:
    """
    Chunk markdown file respecting semantic boundaries.

    Prefers splitting at headers and section boundaries.
    """
    chunks = []
    start = 0
    ordinal = 0

    while start < len(text):
        target_end = start + size
        if target_end >= len(text):
            # Last chunk
            chunk_text = text[start:]
            chunks.append(Chunk(
                text=chunk_text,
                ordinal=ordinal,
                char_start=start,
                char_end=len(text),
            ))
            break

        # Find best split point
        split_pos = find_best_split(text, target_end, size)

        chunk_text = text[start:split_pos]
        chunks.append(Chunk(
            text=chunk_text,
            ordinal=ordinal,
            char_start=start,
            char_end=split_pos,
        ))
        ordinal += 1

        # Next chunk starts at the boundary (no overlap for markdown)
        # This ensures clean splits at headers
        start = split_pos

    return chunks


def extract_python_boundaries(text: str) -> list[int]:
    """
    Extract character positions where top-level definitions start.

    Boundary rules:
    - Decorators stay with their function/class
    - Module docstrings are separate
    - Only top-level definitions (nested excluded)

    Returns list of character positions sorted ascending.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Invalid Python, return empty
        return []

    boundaries = []
    lines = text.splitlines(keepends=True)

    # Calculate character positions for each line
    line_positions = [0]
    for line in lines:
        line_positions.append(line_positions[-1] + len(line))

    # Only iterate top-level nodes from module body
    if hasattr(tree, 'body'):
        for node in tree.body:
            # Top-level function or class definitions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if hasattr(node, 'lineno') and node.lineno > 0:
                    # Get decorator positions if any
                    if node.decorator_list:
                        first_decorator = node.decorator_list[0]
                        if hasattr(first_decorator, 'lineno') and first_decorator.lineno > 0:
                            char_pos = line_positions[first_decorator.lineno - 1]
                            boundaries.append(char_pos)
                    else:
                        char_pos = line_positions[node.lineno - 1]
                        boundaries.append(char_pos)

    return sorted(set(boundaries))


def chunk_python_file(text: str, max_chars: int = 3000) -> list[Chunk]:
    """
    Chunk Python file at function/class boundaries using AST.

    Falls back to generic chunking if AST parsing fails.
    """
    boundaries = extract_python_boundaries(text)

    if not boundaries:
        # Fallback to generic chunking
        return chunk_generic_file(text, size=max_chars)

    # Always start with position 0
    if boundaries[0] != 0:
        boundaries = [0] + boundaries

    chunks = []
    ordinal = 0
    i = 0

    while i < len(boundaries):
        start = boundaries[i]
        # Find the last boundary that keeps chunk < max_chars
        j = i + 1
        while j < len(boundaries) and boundaries[j] - start < max_chars:
            j += 1
        # j now points to first boundary that would exceed max_chars, or past end
        # Use j-1 as the end boundary (or end of text)
        if j - 1 > i:
            # Multiple boundaries fit in max_chars, use the last one
            end = boundaries[j - 1]
            i = j - 1
        elif j < len(boundaries):
            # Next boundary exceeds max_chars, include it anyway to avoid tiny chunks
            end = boundaries[j]
            i = j
        else:
            # No more boundaries
            end = len(text)
            i = len(boundaries)

        chunk_text = text[start:end]
        chunks.append(Chunk(
            text=chunk_text,
            ordinal=ordinal,
            char_start=start,
            char_end=end,
        ))
        ordinal += 1

    if not chunks:
        chunks.append(Chunk(text=text, ordinal=0, char_start=0, char_end=len(text)))

    return chunks
