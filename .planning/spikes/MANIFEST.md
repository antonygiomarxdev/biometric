# Spike Manifest

## Idea
Cleanup spike — analyze the biometric codebase to identify unused files, dead code, Spanish names/comments that should be English, and redundant/duplicate code. Produce an inventory for the user to review before taking action.

## Requirements
- Code must be in English (identifiers, comments, docstrings, variable names)
- File names should be in English
- Full type annotations, no `Any`/`any`
- Clean Architecture separation of concerns

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | cleanup-inventory | standard | Given the codebase, when analyzed for dead code and naming issues, then a complete inventory is produced for user review | VALIDATED | cleanup, naming, dead-code, spanish |
| 002 | llm-forensic-security | standard | Estrategias de privacidad y gateway multi-proveedor seguro | VALIDATED | security, llm |
