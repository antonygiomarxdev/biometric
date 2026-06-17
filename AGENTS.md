# Project Conventions

## Language

- **UI/UX:** Spanish (users in Nicaragua)
- **Code:** English — all identifiers, comments, docstrings, variable names, function names, class names
- **File names:** English
- **Commit messages:** English (conventional commits)
- **Planning artifacts:** English (PLAN.md, SUMMARY.md, etc.)
- **Error messages in API responses:** Spanish (user-facing)

## Stack

- Backend: Python 3.12+ / FastAPI
- Frontend: React + TypeScript + Vite
- Database: PostgreSQL + Qdrant
- Storage: MinIO
- Auth: JWT + bcrypt

## Commit Convention

```
<type>(<scope>): <description>
```

Types: feat, fix, docs, refactor, test, chore
Scope: phase-plan number (e.g., 01-02)

## Architecture

- **Clean Architecture / Hexagonal:** Domain logic separated from infrastructure. Use repository pattern for data access, services for business logic, routers/controllers for HTTP concerns.
- **Dependency Injection:** FastAPI `Depends()` for DI wiring. Never hardcode dependencies.
- **Separation of concerns:** Each layer knows only about the layer directly inside it. Routers → Services → Repositories → Models.
- **Clean Code:** Meaningful names, small focused functions, no commented-out code, DRY, single responsibility. Every function does one thing.
- **Acceptance criteria for all code:** Must follow Clean Architecture + Clean Code. Code review gate checks these.

## Code Quality

- **Strict typing:** Full type annotations everywhere. No `Any`, no `object`, no loose `dict`/`list` without generic params.
- **Python:** Use `pyright` strict mode. All function signatures typed, all class attributes typed.
- **TypeScript:** `strict: true` in tsconfig. No `any`, no `// @ts-ignore`, no `as any`.
