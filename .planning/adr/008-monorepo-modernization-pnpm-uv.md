# ADR 008: Modernización de Monorepo y Build Systems (pnpm + uv)

## Estado
Aceptado

## Contexto
El repositorio presentaba severos problemas estructurales que impedían el desarrollo determinista y seguro:
1. **Infierno de Entornos (PEP 668):** Sistemas operativos modernos basados en Debian bloquean la instalación de paquetes con `pip` a nivel de sistema. Se requerían entornos virtuales complejos de administrar.
2. **Monorepo Incoherente:** Existía la carpeta `node_modules` en la raíz junto a un `package-lock.json`, pero no existía un `package.json` raíz declarando los workspaces. Además, `turbo.json` estaba corrupto (contenía código hexadecimal de una imagen PostScript).
3. **Build Backend Obsoleto:** El backend de Python utilizaba `setuptools` que es lento y genera basura residual (`.egg-info`) directamente en los directorios de código fuente (`src/`).
4. **Falta de Lockfile Seguro:** Aunque el backend usaba grupos de dependencias de `pyproject.toml`, no generaba un lockfile, lo que provocaba descargas impredecibles de versiones secundarias en producción.

## Decisión
Hemos decidido migrar el repositorio a los estándares más rápidos y estrictos de la industria:
1. **Gestor de Paquetes JS (pnpm):** Se implementó `pnpm` (junto con `pnpm-workspace.yaml`) para manejar el monorepo y el frontend por su eficiencia en disco y estrictez en los módulos.
2. **Gestor de Paquetes Python (uv):** Se adoptó `uv` de Astral como gestor unificado para el backend. Resuelve dependencias en milisegundos e ignora las limitaciones restrictivas de PEP 668 del OS usando su propia cadena de herramientas escrita en Rust.
3. **Gestor de Tareas (Turborepo):** Se corrigió y restauró la configuración de pipelines paralelos (`turbo.json`).
4. **Build Backend (Hatchling):** Se migró `pyproject.toml` de `setuptools` a `hatchling` para un empaquetado de ruedas (wheels) sin basura.

## Consecuencias
*   **Positivas:**
    *   **Velocidad Extrema:** La instalación y resolución de dependencias de Python y JS ahora toman milisegundos en lugar de minutos.
    *   **Determinismo Absoluto:** Ambos ecosistemas tienen sus lockfiles correspondientes (`pnpm-lock.yaml` y `uv.lock`) garantizando builds idénticos entre desarrollo, CI y producción.
    *   **Repositorio Limpio:** Ya no hay carpetas basura (`.egg-info`, `__pycache__`) regadas por el código fuente.
*   **Negativas / Curva de aprendizaje:**
    *   Los desarrolladores deben aprender a usar `uv run` en lugar de activar manualmente un entorno virtual con `source .venv/bin/activate`.
    *   Los comandos de JS ahora usan `pnpm` en lugar de `npm`.