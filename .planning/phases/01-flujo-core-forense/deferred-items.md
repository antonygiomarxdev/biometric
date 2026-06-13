# Deferred Items — Phase 01-08 Execution

Pre-existing build issues discovered during plan 01-08 execution that
block the full `npm run build` but are unrelated to this plan's changes:

1. **Missing dependency `@radix-ui/react-dropdown-menu`** — `dropdown-menu.tsx`
   imports this package but it is not in `package.json` or installed. This
   component was already in the codebase before this plan.

2. **`enum` keyword in `logger.ts`** — The project uses `erasableSyntaxOnly: true`
   in `tsconfig.app.json` (TypeScript 5.8+), which forbids `enum` because it
   requires runtime emit. `logger.ts` uses `export enum LogLevel`. This file is
   pre-existing.

Both issues predate this plan. Either should be fixed by a maintainer in a
dedicated cleanup pass.
