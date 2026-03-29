import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ChevronRight,
  Moon,
  Settings2,
  Sparkles,
  Sun,
  Terminal,
  X,
} from 'lucide-react';

const jsonModules = import.meta.glob('../data/*.json', { eager: true });
const jsonlModules = import.meta.glob('../data/*.jsonl', {
  eager: true,
  query: '?raw',
  import: 'default',
});

const dataModules = {
  ...jsonModules,
  ...jsonlModules,
};

function parseJsonl(rawData) {
  if (typeof rawData !== 'string') {
    return [];
  }

  return rawData
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

function normalizeQuotes(rawData) {
  const source = rawData?.default ?? rawData;
  const list = typeof source === 'string' ? parseJsonl(source) : source;

  if (!Array.isArray(list)) {
    return [];
  }

  return list
    .map((entry) => {
      if (typeof entry === 'string') {
        return entry.trim();
      }
      if (entry && typeof entry === 'object') {
        const value = entry.output ?? entry.cleaned ?? entry.text ?? '';
        return String(value).trim();
      }
      return '';
    })
    .filter((item) => item.length > 0);
}

function getFileLabel(filePath) {
  const parts = filePath.split('/');
  return parts[parts.length - 1];
}

function pickRandom(list) {
  if (!list.length) {
    return '';
  }
  const index = Math.floor(Math.random() * list.length);
  return list[index];
}

export default function App() {
  const files = useMemo(() => {
    const entries = Object.entries(dataModules)
      .map(([path, data]) => {
        const quotes = normalizeQuotes(data);
        return {
          path,
          label: getFileLabel(path),
          quotes,
        };
      })
      .filter((item) => item.quotes.length > 0)
      .sort((a, b) => a.label.localeCompare(b.label, 'de'));

    return entries;
  }, []);

  const defaultFile =
    files.find((f) => f.label === 'cleaned_deduplicated.json')?.path ??
    files[0]?.path ??
    '';

  const [selectedFile, setSelectedFile] = useState(defaultFile);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [quoteVersion, setQuoteVersion] = useState(0);
  const [mode, setMode] = useState(() => {
    const saved = localStorage.getItem('ui-mode');
    if (saved === 'light' || saved === 'dark') {
      return saved;
    }
    return 'dark';
  });
  const [promptInput, setPromptInput] = useState('Gib mir einen Trinkspruch');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationError, setGenerationError] = useState('');
  const [generationLogs, setGenerationLogs] = useState('');
  const logRef = useRef(null);

  const selectedQuotes = useMemo(() => {
    return files.find((file) => file.path === selectedFile)?.quotes ?? [];
  }, [files, selectedFile]);

  const [quote, setQuote] = useState(() => pickRandom(selectedQuotes));

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', mode);
    localStorage.setItem('ui-mode', mode);
  }, [mode]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [generationLogs]);

  function nextQuote() {
    if (!selectedQuotes.length) {
      setQuote('Keine Trinksprüche in dieser Datei gefunden.');
      return;
    }

    let next = pickRandom(selectedQuotes);
    if (selectedQuotes.length > 1 && next === quote) {
      next = pickRandom(selectedQuotes);
    }

    setQuote(next);
    setQuoteVersion((v) => v + 1);
  }

  function onFileChange(event) {
    const file = event.target.value;
    setSelectedFile(file);

    const quotes = files.find((item) => item.path === file)?.quotes ?? [];
    setQuote(pickRandom(quotes));
    setQuoteVersion((v) => v + 1);
  }

  async function generateFromModel() {
    const prompt = promptInput.trim() || 'Gib mir einen Trinkspruch';
    setIsGenerating(true);
    setGenerationError('');
    setGenerationLogs('Starte Python-Generierung...\n');

    try {
      const response = await fetch('/api/generate-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.error || 'Generierung fehlgeschlagen.');
      }

      if (!response.body) {
        throw new Error('Streaming wird vom Browser nicht unterstützt.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffered = '';
      let generated = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffered += decoder.decode(value, { stream: true });
        const lines = buffered.split('\n');
        buffered = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.trim()) {
            continue;
          }

          let event;
          try {
            event = JSON.parse(line);
          } catch {
            setGenerationLogs((prev) => (prev + line + '\n').slice(-50000));
            continue;
          }

          if (event.type === 'log') {
            const chunk = String(event.chunk || '');
            setGenerationLogs((prev) => (prev + chunk).slice(-50000));
          }

          if (event.type === 'result') {
            generated = String(event.output || '').trim();
          }

          if (event.type === 'error') {
            throw new Error(event.error || 'Generierung fehlgeschlagen.');
          }
        }
      }

      if (!generated) {
        throw new Error('Leere Antwort vom Modell erhalten.');
      }

      setQuote(generated);
      setQuoteVersion((v) => v + 1);
      setGenerationLogs((prev) => `${prev}\nFertig.\n`);
    } catch (error) {
      setGenerationError(
        error.message || 'Unbekannter Fehler bei der Generierung.',
      );
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="background-glow" />
      <button
        className={`settings-backdrop ${settingsOpen ? 'open' : ''}`}
        aria-label="Settings schließen"
        onClick={() => setSettingsOpen(false)}
      />

      <header className="top-bar">
        <button
          className="mode-btn"
          onClick={() => setMode((m) => (m === 'dark' ? 'light' : 'dark'))}
        >
          {mode === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          <span>{mode === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
        </button>

        <button
          className="settings-btn"
          onClick={() => setSettingsOpen((s) => !s)}
        >
          <Settings2 size={16} />
          <span>Einstellungen</span>
        </button>
      </header>

      <main className="quote-stage">
        <p className="eyebrow">Trinkspruch des Tages</p>

        <blockquote key={quoteVersion} className="quote-card">
          <span className="quote-mark" aria-hidden="true">
            “
          </span>
          <p className="quote-text">
            {quote || 'Keine Daten gefunden. Lege eine JSON-Datei in data/ ab.'}
          </p>
        </blockquote>

        <div className="controls-grid">
          <button className="next-btn" onClick={nextQuote}>
            <ChevronRight size={18} />
            <span>Nächster Trinkspruch</span>
          </button>

          <div className="generator-box">
            <label htmlFor="promptInput">Direkt generieren</label>
            <div className="generator-row">
              <input
                id="promptInput"
                value={promptInput}
                onChange={(e) => setPromptInput(e.target.value)}
                placeholder="z. B. Eleganter Trinkspruch zum Geburtstag"
                disabled={isGenerating}
              />
              <button
                className="generate-btn"
                onClick={generateFromModel}
                disabled={isGenerating}
              >
                <Sparkles
                  size={16}
                  className={isGenerating ? 'icon-spin' : ''}
                />
                <span>{isGenerating ? 'Generiert...' : 'Generieren'}</span>
              </button>
            </div>
            {generationError ? (
              <p className="error-text">{generationError}</p>
            ) : null}
            <p className="log-title">
              <Terminal size={14} />
              <span>Python Live-Log</span>
            </p>
            <pre ref={logRef} className="log-view" aria-live="polite">
              {generationLogs || 'Noch keine Python-Logs.'}
            </pre>
          </div>
        </div>
      </main>

      <aside className={`settings-panel ${settingsOpen ? 'open' : ''}`}>
        <div className="settings-head">
          <h2>Settings</h2>
          <button
            className="close-settings-btn"
            aria-label="Settings schließen"
            onClick={() => setSettingsOpen(false)}
          >
            <X size={18} />
          </button>
        </div>
        <label htmlFor="fileSelect">JSON-Datei aus data/</label>
        <select id="fileSelect" value={selectedFile} onChange={onFileChange}>
          {files.map((file) => (
            <option key={file.path} value={file.path}>
              {file.label}
            </option>
          ))}
        </select>

        <p className="settings-meta">{selectedQuotes.length} Sprüche geladen</p>
      </aside>
    </div>
  );
}
