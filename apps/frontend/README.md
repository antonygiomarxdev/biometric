# Biometric Frontend

Aplicación cliente para analistas y peritos forenses. Desarrollada con **React, TypeScript y Vite**.

## 🚀 Módulos del Cliente

1. **Área de Captura e Ingreso (Enrollment)**
   - Interfaz de escaneo/subida de huellas dactilares decadactilares (conocidas).
2. **Área de Búsqueda Latente (Search)**
   - Subida de fragmentos o huellas levantadas de la escena del crimen.
   - Envía el fragmento al motor de *RAG Dactilar* del backend.
3. **Mesa de Cotejo (Comparison View)**
   - Workspace visual donde el perito puede ver superpuestas la huella latente y el hit sugerido por la IA/Matemática.
   - Herramientas de visualización de minucias y líneas base.
4. **Módulo de Resoluciones y Burocracia**
   - Panel para emitir el veredicto oficial (Match / Exclusión / Inconcluso).
   - Botón mágico para generar el Dictamen Pericial Legal en formato PDF mediante GenAI.

## Stack Tecnológico

- **Core:** React 18 + TypeScript
- **Build Tool:** Vite + SWC (Fast Refresh)
- **Styling:** Tailwind CSS / CSS Modules
- **State Management:** React Query / Zustand

## Desarrollo

```bash
# Instalar dependencias
pnpm install

# Correr servidor de desarrollo
pnpm run dev

# Compilar para producción
pnpm run build
```
