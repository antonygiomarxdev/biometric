# 🚀 Setup con uv - Guía Rápida

## ✅ Configuración Automática

Este proyecto está configurado para usar **`uv`** como gestor de dependencias y entornos virtuales.

---

## 📋 Requisitos Previos

1. **Python 3.12+** instalado
2. **uv** instalado:

```powershell
# Opción 1: pip
pip install uv

# Opción 2: winget (Windows)
winget install astral-sh.uv

# Opción 3: PowerShell (Windows)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verificar instalación:
```powershell
uv --version
```

---

## 🔧 Setup Inicial (Una Vez)

### Opción A: Script Automático (Recomendado)

```powershell
.\setup.ps1
```

Este script:
- ✅ Verifica que `uv` esté instalado
- ✅ Sincroniza todas las dependencias
- ✅ Crea el entorno virtual `.venv`
- ✅ Verifica que todo funcione

### Opción B: Manual

```powershell
# 1. Sincronizar dependencias (crea .venv automáticamente)
uv sync

# 2. Verificar que funciona
uv run python --version
```

---

## 🎯 Activación del Entorno Virtual

### En PowerShell (Recomendado)

```powershell
# Activar entorno
. .\activate.ps1

# El script automáticamente:
# - Ejecuta `uv sync` si es necesario
# - Activa el entorno virtual
# - Muestra el Python que está usando
```

### En CMD

```cmd
activate.bat
```

### En VS Code/Cursor

**¡Ya está configurado!** El entorno se activa automáticamente cuando abres el proyecto.

El archivo `.vscode/settings.json` está configurado para:
- ✅ Usar el Python de `.venv` automáticamente
- ✅ Activar el entorno en terminales nuevas
- ✅ Configurar pytest correctamente

---

## 📦 Comandos Útiles con uv

### Sincronizar dependencias
```powershell
uv sync                    # Sincroniza según pyproject.toml
uv sync --dev             # Incluye dependencias de desarrollo
```

### Agregar nuevas dependencias
```powershell
uv add numpy              # Agrega a pyproject.toml y sincroniza
uv add --dev pytest       # Agrega como dependencia de desarrollo
```

### Ejecutar comandos en el entorno
```powershell
uv run python script.py   # Ejecuta en el entorno virtual
uv run pytest             # Ejecuta pytest
uv run python test_socofing.py
```

### Actualizar dependencias
```powershell
uv sync --upgrade         # Actualiza todas las dependencias
```

---

## 🔄 Flujo de Trabajo Diario

### Primera vez en el proyecto

```powershell
# 1. Clonar/abrir proyecto
cd biometric

# 2. Setup inicial (una vez)
.\setup.ps1

# 3. Activar entorno
. .\activate.ps1
```

### Día a día

```powershell
# Solo activar el entorno
. .\activate.ps1

# O ejecutar comandos directamente
uv run python test_socofing.py
```

---

## 🎮 En VS Code/Cursor

### Activación Automática

El entorno se activa **automáticamente** gracias a `.vscode/settings.json`:

1. Abre el proyecto en VS Code/Cursor
2. Selecciona el intérprete: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Elige: `.venv\Scripts\python.exe`
4. **¡Listo!** Todos los scripts y terminales usarán este entorno

### Terminal Integrada

Cuando abres una terminal en VS Code/Cursor:
- ✅ El entorno se activa automáticamente
- ✅ `python` apunta a `.venv\Scripts\python.exe`
- ✅ Todas las dependencias están disponibles

---

## 🐛 Troubleshooting

### Error: "uv no está instalado"

```powershell
# Instalar uv
pip install uv

# O con winget
winget install astral-sh.uv
```

### Error: "Entorno virtual no encontrado"

```powershell
# Crear el entorno
uv sync
```

### VS Code no detecta el entorno

1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Elegir: `.venv\Scripts\python.exe`
3. Si no aparece, ejecutar `uv sync` primero

### El entorno no se activa automáticamente

Verificar que `.vscode/settings.json` existe y tiene:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"
}
```

---

## 📊 Ventajas de uv

- ⚡ **10-100x más rápido** que pip
- 🔒 **Reproducible** con `uv.lock`
- 🎯 **Simple**: Un solo comando (`uv sync`)
- 🚀 **Moderno**: Reemplaza pip, venv, pip-tools

---

## 🔗 Recursos

- Documentación uv: https://docs.astral.sh/uv/
- GitHub: https://github.com/astral-sh/uv
- PyProject.toml: https://peps.python.org/pep-0621/
