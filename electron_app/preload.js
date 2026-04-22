/**
 * Preload — runs in the renderer sandbox.
 * Exposes a minimal, typed bridge API via contextBridge.
 */
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getBridgeUrl:   ()        => ipcRenderer.invoke('get-bridge-url'),
  onBridgeStatus: (handler) => ipcRenderer.on('bridge-status', (_event, val) => handler(val)),
});
