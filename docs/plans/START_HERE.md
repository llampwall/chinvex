# P0 Implementation - Start Here

## Plan Structure

The P0 implementation plan is split into two parts:

1. **PART 1 - Foundation** (`2026-01-26-p0-PART1-foundation.md`)
   - Tasks 1-10: Schema, fingerprinting, context registry, adapters, chunking
   - Status: ✅ ALL 10 TASKS COMPLETE

2. **PART 2 - Integration** (`2026-01-26-p0-PART2-integration.md`)
   - Tasks 11-18: Scoring, CLI updates, Codex ingestion, MCP tool
   - Status: All TODO (depends on Part 1 completion)

## Current Status

### ✅ Completed (ALL DONE)
- Task 1: Schema Version + Meta Table
- Task 2: source_fingerprints Table
- Task 3: Context Registry Data Structures
- Task 4: CLI Command - context create
- Task 5: CLI Command - context list
- Task 6: Auto-Migration from Old Config
- Task 7: Conversation Chunking with Token Approximation
- Task 8: Codex App-Server Client (Schema Capture)
- Task 9: Codex App-Server Schemas (Pydantic)
- Task 10: Normalize App-Server to ConversationDoc

## Instructions for Executing Agent

1. ✅ **PART 1 COMPLETE** - All 10 foundation tasks done
2. **START HERE:** Open `2026-01-26-p0-PART2-integration.md`
3. Begin with Task 11 (first TODO task in Part 2)
4. Follow TDD steps for each task
5. Continue until Part 2 is complete

## Status Updates

When you complete a task:
1. Mark the task header as `✅ DONE` in the plan file
2. Update the status table at the top of the plan
3. Update the Instructions for Executing Agent
4. Commit with clear message
5. Move to next TODO task

Do NOT skip tasks or work out of order unless dependencies allow.
