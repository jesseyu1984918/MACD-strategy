const { app, BrowserWindow, dialog } = require("electron");
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const { spawn, spawnSync } = require("node:child_process");

const STREAMLIT_HOST = "127.0.0.1";
const STREAMLIT_PORT = 8501;
const STREAMLIT_URL = `http://${STREAMLIT_HOST}:${STREAMLIT_PORT}`;

let mainWindow;
let streamlitProcess;
let isQuitting = false;

function getAppRoot() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "app");
  }
  return app.getAppPath();
}

function commandExists(command) {
  const result = spawnSync("where", [command], {
    stdio: "ignore",
    windowsHide: true
  });
  return result.status === 0;
}

function getPythonCandidates() {
  const root = getAppRoot();
  return [
    path.join(root, "venv", "Scripts", "python.exe"),
    path.join(root, "..", "venv", "Scripts", "python.exe"),
    "py",
    "python"
  ];
}

function getStreamlitScriptPath() {
  return path.join(getAppRoot(), "stock_center.py");
}

function waitForServer(timeoutMs = 30000) {
  const startedAt = Date.now();

  return new Promise((resolve, reject) => {
    const tryConnect = () => {
      const req = http.get(STREAMLIT_URL, (res) => {
        res.resume();
        resolve();
      });

      req.on("error", () => {
        if (Date.now() - startedAt > timeoutMs) {
          reject(new Error("Timed out waiting for Streamlit to start."));
          return;
        }

        setTimeout(tryConnect, 500);
      });
    };

    tryConnect();
  });
}

function stopStreamlit() {
  if (!streamlitProcess || streamlitProcess.killed) {
    return;
  }

  streamlitProcess.kill();
  streamlitProcess = null;
}

async function startStreamlit() {
  const root = getAppRoot();
  const scriptPath = getStreamlitScriptPath();
  const errors = [];

  for (const candidate of getPythonCandidates()) {
    const isExePath = candidate.toLowerCase().endsWith(".exe");
    if (isExePath && !fs.existsSync(candidate)) {
      continue;
    }
    if (!isExePath && !commandExists(candidate)) {
      continue;
    }

    const argsPrefix = candidate === "py" ? ["-3"] : [];
    const args = [
      ...argsPrefix,
      "-m",
      "streamlit",
      "run",
      scriptPath,
      "--server.headless",
      "true",
      "--server.address",
      STREAMLIT_HOST,
      "--server.port",
      String(STREAMLIT_PORT),
      "--browser.gatherUsageStats",
      "false"
    ];

    streamlitProcess = spawn(candidate, args, {
      cwd: root,
      stdio: "pipe",
      windowsHide: true
    });

    streamlitProcess.stdout.on("data", (data) => {
      process.stdout.write(`[streamlit] ${data}`);
    });

    streamlitProcess.stderr.on("data", (data) => {
      process.stderr.write(`[streamlit] ${data}`);
    });

    try {
      await waitForServer();
      return;
    } catch (error) {
      errors.push(`${candidate}: ${error instanceof Error ? error.message : String(error)}`);
      stopStreamlit();
    }
  }

  throw new Error(`Unable to start Streamlit. Tried: ${errors.join(" | ") || "no Python interpreter found"}`);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1100,
    minHeight: 760,
    autoHideMenuBar: true,
    backgroundColor: "#0f172a",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.loadURL(STREAMLIT_URL);
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  try {
    await startStreamlit();
    createWindow();
  } catch (error) {
    dialog.showErrorBox(
      "Stock Pick Panel",
      error instanceof Error ? error.message : "Failed to start the application."
    );
    app.quit();
  }
});

app.on("before-quit", () => {
  isQuitting = true;
  stopStreamlit();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
