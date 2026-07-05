# Electron Packaging

This repo already uses Streamlit for the UI. The Electron app starts a local Streamlit server and opens it in a desktop window.
The Windows package now bundles the project virtual environment so the app can run without requiring a separate Python install on the target machine.

## Prerequisites

- Python 3.12 with the current project dependencies installed
- Node.js 20 or newer

## Install Electron tooling

```powershell
npm install
```

## Run the desktop app in development

```powershell
npm run app:dev
```

## Build a Windows installer

```powershell
npm run app:dist
```

The installer output is written to `dist/`.

## Packaging notes

- The Electron shell loads `stock_center.py`.
- In development it prefers the repo virtual environment at `venv\Scripts\python.exe`.
- If that virtual environment is missing, it falls back to `py -3` and then `python`.
- The packaged app includes the Python source files, CSV assets, and the `venv` directory.
- Because the full virtual environment is bundled, the installer will be much larger than a normal Electron-only app.
- If you later want a smaller installer, the next optimization would be replacing the bundled `venv` with a frozen Python executable.
