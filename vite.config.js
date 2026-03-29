import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { spawn } from 'node:child_process';

function writeEvent(res, payload) {
  res.write(`${JSON.stringify(payload)}\n`);
}

function addGenerateApi(server) {
  server.middlewares.use('/api/generate-stream', (req, res) => {
    if (req.method !== 'POST') {
      res.statusCode = 405;
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ error: 'Method not allowed' }));
      return;
    }

    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
    });

    req.on('end', () => {
      let payload = {};
      try {
        payload = body ? JSON.parse(body) : {};
      } catch {
        res.statusCode = 400;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ error: 'Invalid JSON body' }));
        return;
      }

      const prompt =
        typeof payload.prompt === 'string' && payload.prompt.trim().length > 0
          ? payload.prompt.trim()
          : 'Gib mir einen Trinkspruch';

      const pythonCmd = existsSync(
        join(process.cwd(), '.venv', 'Scripts', 'python.exe'),
      )
        ? join(process.cwd(), '.venv', 'Scripts', 'python.exe')
        : 'python';

      const scriptPath = join(process.cwd(), 'inference', 'generate.py');
      const child = spawn(pythonCmd, [scriptPath, '--prompt', prompt], {
        cwd: process.cwd(),
      });

      let stdout = '';
      let stderr = '';

      res.statusCode = 200;
      res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('X-Accel-Buffering', 'no');

      child.stdout.on('data', (chunk) => {
        const text = String(chunk);
        stdout += text;
        writeEvent(res, { type: 'log', chunk: text });
      });

      child.stderr.on('data', (chunk) => {
        const text = String(chunk);
        stderr += text;
        writeEvent(res, { type: 'log', chunk: text });
      });

      child.on('close', (code) => {
        if (code !== 0) {
          writeEvent(res, {
            type: 'error',
            error: stderr.trim() || `generate.py exited with code ${code}`,
          });
          res.end();
          return;
        }

        const lines = stdout
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter((line) => line.length > 0);
        const output = lines.length > 0 ? lines[lines.length - 1] : '';

        writeEvent(res, { type: 'result', output });
        res.end();
      });
    });
  });
}

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'python-generate-api',
      configureServer(server) {
        addGenerateApi(server);
      },
      configurePreviewServer(server) {
        addGenerateApi(server);
      },
    },
  ],
});
