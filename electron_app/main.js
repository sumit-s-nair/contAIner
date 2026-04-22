/**
 * Electron main process.
 *
 * Responsibilities:
 *  1. Spawn the Python Flask bridge server (running in WSL via `wsl` command)
 *  2. Wait until the bridge /health endpoint is ready
 *  3. Open the BrowserWindow pointing at renderer/index.html
 *  4. Kill the bridge process on app quit
 */

const { app, BrowserWindow, ipcMain, shell } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

// ── Config ────────────────────────────────────────────────────────────────────
const BRIDGE_PORT = 5050;
const BRIDGE_URL  = `http://localhost:${BRIDGE_PORT}`;
const HEALTH_URL  = `${BRIDGE_URL}/health`;

// WSL path to the project — adjust if your WSL username differs
const WSL_PROJECT_ROOT = '/home/sumit/contAIner';
const WSL_CONDA_ENV    = 'pyg-pip';
const BRIDGE_MODULE_CMD = `cd "${WSL_PROJECT_ROOT}" && conda run -n ${WSL_CONDA_ENV} --no-capture-output python -m electron_app.bridge.server`;

let bridgeProcess = null;
let mainWindow    = null;

// ── Spawn Flask bridge in WSL ─────────────────────────────────────────────────
function startBridge() {
  console.log('[main] Spawning Python bridge in WSL...');
  bridgeProcess = spawn('wsl', ['bash', '-c', BRIDGE_MODULE_CMD], {
    detached: false,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  bridgeProcess.stdout.on('data', d => process.stdout.write(`[bridge] ${d}`));
  bridgeProcess.stderr.on('data', d => process.stderr.write(`[bridge-err] ${d}`));
  bridgeProcess.on('exit', (code) => {
    console.warn(`[main] Bridge exited with code ${code}`);
  });
}

// ── Poll /health until bridge is ready ───────────────────────────────────────
function waitForBridge(maxAttempts = 60, intervalMs = 1000) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    const check = () => {
      attempts++;
      const req = http.get(HEALTH_URL, (res) => {
        if (res.statusCode === 200) {
          console.log('[main] Bridge is ready ✓');
          resolve();
        } else {
          retry();
        }
        res.resume();
      });
      req.on('error', retry);
      req.setTimeout(800, () => { req.destroy(); retry(); });
    };
    const retry = () => {
      if (attempts >= maxAttempts) {
        reject(new Error('Bridge did not become ready in time'));
      } else {
        setTimeout(check, intervalMs);
      }
    };
    check();
  });
}

// ── Create the BrowserWindow ──────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 960,
    height: 720,
    minWidth: 720,
    minHeight: 540,
    backgroundColor: '#0f111a',
    titleBarStyle: 'hiddenInset',
    frame: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  mainWindow.on('closed', () => { mainWindow = null; });
}

// ── App lifecycle ─────────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  startBridge();

  // Show a minimal loading window immediately while waiting
  createWindow();
  mainWindow.webContents.on('did-finish-load', () => {
    mainWindow.webContents.send('bridge-status', { ready: false, message: 'Loading AI models...' });
  });

  try {
    await waitForBridge();
    if (mainWindow) {
      mainWindow.webContents.send('bridge-status', { ready: true });
    }
  } catch (err) {
    console.error('[main] Bridge failed to start:', err.message);
    if (mainWindow) {
      mainWindow.webContents.send('bridge-status', {
        ready: false,
        message: 'Failed to start Python bridge. Check WSL and conda env.',
      });
    }
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
  if (bridgeProcess) {
    console.log('[main] Killing bridge process...');
    bridgeProcess.kill('SIGTERM');
  }
});

// ── IPC: pass BRIDGE_URL to renderer ─────────────────────────────────────────
ipcMain.handle('get-bridge-url', () => BRIDGE_URL);
