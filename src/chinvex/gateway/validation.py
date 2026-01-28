"""Request validation models."""

import re
from typing import Optional
from pydantic import BaseModel, field_validator

CONTEXT_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,50}$')
CHUNK_ID_PATTERN = re.compile(r'^[a-f0-9]{12}$')


class EvidenceRequest(BaseModel):
    """Request for /v1/evidence endpoint."""
    context: str
    query: str
    k: int = 8
    source_types: Optional[list[str]] = None

    @field_validator('context')
    def validate_context(cls, v):
        if not CONTEXT_PATTERN.match(v):
            raise ValueError('Invalid context name format')
        return v

    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 1000:
            raise ValueError('Query exceeds 1000 character limit')
        if '\x00' in v:
            raise ValueError('Null bytes not allowed in query')
        return v.strip()

    @field_validator('k')
    def validate_k(cls, v):
        if v < 1 or v > 20:
            raise ValueError('k must be between 1 and 20')
        return v


class SearchRequest(BaseModel):
    """Request for /v1/search endpoint."""
    context: str
    query: str
    k: int = 10
    source_types: Optional[list[str]] = None
    no_recency: bool = False

    @field_validator('context')
    def validate_context(cls, v):
        if not CONTEXT_PATTERN.match(v):
            raise ValueError('Invalid context name format')
        return v

    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 1000:
            raise ValueError('Query exceeds 1000 character limit')
        if '\x00' in v:
            raise ValueError('Null bytes not allowed in query')
        return v.strip()

    @field_validator('k')
    def validate_k(cls, v):
        if v < 1 or v > 20:
            raise ValueError('k must be between 1 and 20')
        return v


class ChunksRequest(BaseModel):
    """Request for /v1/chunks endpoint."""
    context: str
    chunk_ids: list[str]

    @field_validator('context')
    def validate_context(cls, v):
        if not CONTEXT_PATTERN.match(v):
            raise ValueError('Invalid context name format')
        return v

    @field_validator('chunk_ids')
    def validate_chunk_ids(cls, v):
        if len(v) > 20:
            raise ValueError('Maximum 20 chunk IDs per request')
        return v


class AnswerRequest(BaseModel):
    """Request for /v1/answer endpoint (optional)."""
    context: str
    query: str
    k: int = 8
    grounded: bool = True

    @field_validator('context')
    def validate_context(cls, v):
        if not CONTEXT_PATTERN.match(v):
            raise ValueError('Invalid context name format')
        return v

    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 1000:
            raise ValueError('Query exceeds 1000 character limit')
        return v.strip()

    @field_validator('k')
    def validate_k(cls, v):
        if v < 1 or v > 20:
            raise ValueError('k must be between 1 and 20')
        return v
