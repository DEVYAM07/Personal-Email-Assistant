import { useEffect, useRef, useState } from "react";
import {
  Bot,
  ChevronDown,
  ChevronUp,
  FileText,
  LogOut,
  Mail,
  MessageCircle,
  RefreshCw,
  Search,
  Send,
  Sparkles,
  Trophy,
  User,
  Wifi,
  WifiOff,
} from "lucide-react";

const API_BASE = (import.meta.env.VITE_API_URL || (import.meta.env.PROD ? "https://DarkByteX-personal-email-assistant.hf.space" : "http://localhost:8000")).replace(/\/$/, "");
const HEALTH_URL = `${API_BASE}/api/health`;
const QUERY_URL = `${API_BASE}/api/query`;
const SYNC_URL = `${API_BASE}/api/sync`;
const SYNC_STATUS_URL = `${API_BASE}/api/sync/status`;
const AUTH_LOGIN_URL = `${API_BASE}/api/auth/login`;
const AUTH_STATUS_URL = `${API_BASE}/api/auth/status`;

// helper to detect network/CORS failures vs API errors
function isNetworkErrorMessage(msg) {
  const m = String(msg).toLowerCase();
  return m.includes("failed to fetch") || m.includes("networkerror") || m.includes("net::err_failed") || m.includes("err_failed") || m.includes("load failed") || m.includes("cors") || m.includes("unreachable");
}

const PRESET_QUERIES = [
  { label: "When is my next contest?", icon: Trophy },
  { label: "Find recent invoices", icon: FileText },
  { label: "Summarize unread emails", icon: Mail },
  { label: "What did I miss this week?", icon: Search },
];

// ---------- helpers ----------
function formatDate(dateStr) {
  if (!dateStr) return "";
  try {
    const d = new Date(dateStr);
    if (Number.isNaN(d.getTime())) return String(dateStr);
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return String(dateStr);
  }
}

function unescapeHTML(text) {
  if (!text) return "";
  try {
    // textarea is the safest way to unescape entities in browser
    const txt = document.createElement("textarea");
    txt.innerHTML = text;
    return txt.value;
  } catch {
    return text;
  }
}

export default function App() {
  // health check
  const [health, setHealth] = useState("checking"); // checking | online | offline
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncMessage, setSyncMessage] = useState(null);
  const [messages, setMessages] = useState([]); // [{role, text, sources}]
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedSources, setExpandedSources] = useState({});

  // --- Auth state management per spec ---
  const [userEmail, setUserEmail] = useState(() => {
    try {
      return localStorage.getItem("userEmail") || null;
    } catch {
      return null;
    }
  });
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const endRef = useRef(null);
  const inputRef = useRef(null);

  // ---- health check on mount ----
  const checkHealth = async () => {
    setHealth("checking");
    setError(null);
    try {
      const res = await fetch(HEALTH_URL, {
        method: "GET",
        mode: "cors",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
      // try to parse but not required
      await res.json().catch(() => null);
      setHealth("online");
    } catch (e) {
      setHealth("offline");
      const msg = e instanceof Error ? e.message : String(e);
      if (isNetworkErrorMessage(msg)) {
        setError(`FastAPI is unreachable at ${API_BASE} — is the backend running? (CORS/network error: ${msg})`);
      } else {
        setError(`FastAPI is unreachable at ${API_BASE} — is the backend running?`);
      }
    }
  };

  useEffect(() => {
    checkHealth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- Auth: inspect URL query params for auth=success & email on mount ----
  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      const auth = params.get("auth");
      const email = params.get("email");
      if (auth === "success" && email) {
        // save email to localStorage
        try {
          localStorage.setItem("userEmail", email);
        } catch {}
        setUserEmail(email);
        setIsAuthenticated(true);
        // clear query parameters from browser address bar
        params.delete("auth");
        params.delete("email");
        const newSearch = params.toString();
        const newUrl = window.location.pathname + (newSearch ? `?${newSearch}` : "") + window.location.hash;
        window.history.replaceState({}, "", newUrl);
        return;
      }
    } catch {}
    // Fallback: check localStorage and verify with backend
    try {
      const stored = localStorage.getItem("userEmail");
      if (stored) {
        setUserEmail(stored);
        // verify with backend status route
        fetch(`${AUTH_STATUS_URL}?email=${encodeURIComponent(stored)}`, { mode: "cors" })
          .then((res) => {
            if (!res.ok) throw new Error(`Status ${res.status}`);
            return res.json();
          })
          .then((data) => {
            setIsAuthenticated(!!data.authenticated);
          })
          .catch(() => {
            // if status check fails (network/CORS), assume not authenticated but keep email for UI fallback
            // We set false to show Connect button; avoid generic unreachable banner here
            setIsAuthenticated(false);
          });
      } else {
        setIsAuthenticated(false);
      }
    } catch {
      setIsAuthenticated(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-verify auth status when userEmail changes (except initial success case handled above)
  useEffect(() => {
    if (!userEmail) {
      setIsAuthenticated(false);
      return;
    }
    // Only verify if not already set from URL success; still verify to ensure token valid
    // Avoid double fetch on initial mount where we already verified
    // We can still fetch to confirm
    fetch(`${AUTH_STATUS_URL}?email=${encodeURIComponent(userEmail)}`, { mode: "cors" })
      .then((res) => {
        if (!res.ok) throw new Error(`Status ${res.status}`);
        return res.json();
      })
      .then((data) => setIsAuthenticated(!!data.authenticated))
      .catch(() => setIsAuthenticated(false));
  }, [userEmail]);

  const handleConnect = () => {
    window.location.href = `${API_BASE}/api/auth/login`;
  };

  const handleDisconnect = () => {
    try {
      localStorage.removeItem("userEmail");
    } catch {}
    setUserEmail(null);
    setIsAuthenticated(false);
  };

  // ---- auto-scroll to newest message ----
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading, error]);

  // ---- sync inbox (async polling to avoid Render 50s gateway timeout) ----
  const sendSync = async () => {
    if (!isAuthenticated) {
      setError("Connect your Gmail account to start querying emails");
      return;
    }
    if (isSyncing) return;
    if (health === "offline") {
      console.warn("Sync attempted while health is offline - attempting anyway");
    }
    setIsSyncing(true);
    setSyncMessage(null);
    setError(null);

    // Helper to handle network/CORS vs API errors
    const handleSyncError = (msg) => {
      const isNetwork = isNetworkErrorMessage(msg);
      if (isNetwork) {
        if (health === "online") {
          setError(
            `Network error: Unable to reach ${API_BASE} due to network/CORS. Please ensure the backend allows requests from ${window.location.origin} and is running. (${msg})`
          );
        } else {
          setError(`Network error: FastAPI is unreachable. Please ensure the backend is running on ${API_BASE} (${msg})`);
        }
      } else {
        if (msg.includes("401") || msg.toLowerCase().includes("not authenticated") || msg.toLowerCase().includes("unauthorized")) {
          setError(`Sync failed: Not authenticated. Please reconnect Gmail. (${msg})`);
        } else {
          setError(`Sync failed: ${msg}`);
        }
      }
      setIsSyncing(false);
    };

    try {
      const syncUrl = userEmail ? `${SYNC_URL}?email=${encodeURIComponent(userEmail)}` : SYNC_URL;
      const res = await fetch(syncUrl, {
        method: "POST",
        mode: "cors",
        headers: {
          "Content-Type": "application/json",
          ...(userEmail ? { "X-User-Email": userEmail } : {}),
        },
        body: JSON.stringify(userEmail ? { email: userEmail } : {}),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        let detail = text;
        try {
          const j = JSON.parse(text);
          detail = j.detail || j.message || text;
        } catch {}
        throw new Error(detail || `Sync failed (${res.status})`);
      }
      const data = await res.json();

      // Detect async job response (202 or job_id present) vs legacy synchronous success
      const jobId = data.job_id || data.jobId || data.id || null;
      const isAsync = res.status === 202 || jobId || data.status === "started" || data.status === "pending" || data.status === "running";

      if (isAsync && jobId) {
        // Poll GET /api/sync/status?job_id=...
        setSyncMessage(`Sync started — fetching inbox...`);
        let attempts = 0;
        const maxAttempts = 90; // 3 minutes (90 * 2s)
        let pollTimeout = null;

        const cleanup = () => {
          if (pollTimeout) clearTimeout(pollTimeout);
        };

        const poll = async () => {
          attempts += 1;
          try {
            const statusUrl = `${SYNC_STATUS_URL}?job_id=${encodeURIComponent(jobId)}`;
            const sRes = await fetch(statusUrl, {
              method: "GET",
              mode: "cors",
              headers: {
                ...(userEmail ? { "X-User-Email": userEmail } : {}),
              },
            });
            if (!sRes.ok) {
              const t = await sRes.text().catch(() => "");
              let d = t;
              try {
                const j = JSON.parse(t);
                d = j.detail || j.message || t;
              } catch {}
              throw new Error(d || `Status check failed (${sRes.status})`);
            }
            const sData = await sRes.json();
            const sStatus = sData.status;

            if (sStatus === "completed" || sStatus === "success") {
              const added = sData.added ?? 0;
              const total = sData.total_fetched ?? sData.totalFetched ?? 0;
              setSyncMessage(`Synced ${added} new emails${total ? ` (fetched ${total})` : ""}`);
              setTimeout(() => setSyncMessage(null), 4000);
              setIsSyncing(false);
              cleanup();
              return;
            }
            if (sStatus === "failed" || sStatus === "error") {
              const err = sData.error || sData.detail || "Sync failed";
              throw new Error(err);
            }
            // still running/pending/started
            if (sData.progress) {
              setSyncMessage(`Syncing... ${sData.progress}`);
            } else if (sStatus === "running" || sStatus === "pending" || sStatus === "started") {
              setSyncMessage(`Syncing... ${attempts * 2}s elapsed`);
            }
            if (attempts < maxAttempts) {
              pollTimeout = setTimeout(poll, 2000);
            } else {
              throw new Error("Sync polling timeout — please refresh and check again");
            }
          } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            handleSyncError(msg);
            cleanup();
          }
        };
        // start polling after short delay (allow backend to transition to running)
        setTimeout(poll, 1500);
        return; // keep isSyncing true until poll finishes
      }

      if (isAsync && !jobId) {
        // Async but no job_id (fallback poll by email)
        setSyncMessage(`Sync started — fetching inbox...`);
        let attempts = 0;
        const maxAttempts = 90;
        let pollTimeout = null;
        const cleanup = () => { if (pollTimeout) clearTimeout(pollTimeout); };
        const pollByEmail = async () => {
          attempts += 1;
          try {
            const statusUrl = userEmail ? `${SYNC_STATUS_URL}?email=${encodeURIComponent(userEmail)}` : SYNC_STATUS_URL;
            const sRes = await fetch(statusUrl, { method: "GET", mode: "cors", headers: { ...(userEmail ? { "X-User-Email": userEmail } : {}) } });
            if (!sRes.ok) throw new Error(`Status check failed (${sRes.status})`);
            const sData = await sRes.json();
            const sStatus = sData.status;
            if (sStatus === "completed" || sStatus === "success") {
              const added = sData.added ?? 0;
              setSyncMessage(`Synced ${added} new emails`);
              setTimeout(() => setSyncMessage(null), 4000);
              setIsSyncing(false);
              cleanup();
              return;
            }
            if (sStatus === "failed") throw new Error(sData.error || "Sync failed");
            if (sStatus === "idle" && attempts > 2) {
              // still idle after a bit — maybe job not yet created, keep waiting
              setSyncMessage(`Syncing... ${attempts * 2}s elapsed`);
            } else if (sData.progress) {
              setSyncMessage(`Syncing... ${sData.progress}`);
            }
            if (attempts < maxAttempts) {
              pollTimeout = setTimeout(pollByEmail, 2000);
            } else {
              setIsSyncing(false);
              cleanup();
              setError("Sync polling timeout — please refresh");
            }
          } catch (e) {
            handleSyncError(e instanceof Error ? e.message : String(e));
            cleanup();
          }
        };
        setTimeout(pollByEmail, 1500);
        return;
      }

      // Legacy synchronous path (no job_id, status success)
      const added = data.added ?? 0;
      const total = data.total_fetched ?? data.totalFetched ?? 0;
      setSyncMessage(`Synced ${added} new emails${total ? ` (fetched ${total})` : ""}`);
      setTimeout(() => setSyncMessage(null), 4000);
      setIsSyncing(false);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      handleSyncError(msg);
    }
  };

  const toggleSources = (idx) => {
    setExpandedSources((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const sendQuery = async (question) => {
    if (!isAuthenticated) {
      setError("Connect your Gmail account to start querying emails");
      return;
    }
    const q = (question ?? input).trim();
    if (!q || loading) return;

    setError(null);
    setLoading(true);

    // push user message immediately
    setMessages((prev) => [...prev, { role: "user", text: q, sources: [] }]);
    setInput("");

    try {
      const body = { question: q };
      if (userEmail) body.email = userEmail;
      const queryUrl = userEmail ? `${QUERY_URL}?email=${encodeURIComponent(userEmail)}` : QUERY_URL;
      const res = await fetch(queryUrl, {
        method: "POST",
        mode: "cors",
        headers: {
          "Content-Type": "application/json",
          ...(userEmail ? { "X-User-Email": userEmail } : {}),
        },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        let detail = text;
        try {
          const j = JSON.parse(text);
          detail = j.detail || j.message || text;
        } catch {}
        // Provide status-aware message
        if (res.status === 401) throw new Error(detail || "Unauthorized - Please connect Gmail via /api/auth/login");
        if (res.status === 400) throw new Error(detail || `Bad request (${res.status})`);
        throw new Error(detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      const answer = data.answer ?? data.response ?? data.text ?? "";
      const sources = Array.isArray(data.sources) ? data.sources : Array.isArray(data.references) ? data.references : [];

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: String(answer), sources },
      ]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      // network error vs API error -> show inline error bubble
      const isNetwork = isNetworkErrorMessage(msg);
      let errorBanner;
      let bubbleText;
      if (isNetwork) {
        if (health === "online") {
          errorBanner = `Network error: Unable to reach ${API_BASE} due to network/CORS. Please ensure the backend allows requests from ${window.location.origin} and is running. (${msg})`;
          bubbleText = `⚠️ Could not reach the email service due to network/CORS. Please check that FastAPI allows CORS from ${window.location.origin} and try again.`;
        } else {
          errorBanner = `Network error: FastAPI is unreachable. Please ensure the backend is running on ${API_BASE} (${msg})`;
          bubbleText = `⚠️ Could not reach the email service. Please check that FastAPI is running on ${API_BASE} and try again.`;
        }
      } else {
        // Handle auth vs generic errors without showing unreachable
        if (msg.toLowerCase().includes("not authenticated") || msg.includes("401") || msg.toLowerCase().includes("unauthorized")) {
          errorBanner = `Request failed: Not authenticated. Please connect Gmail. (${msg})`;
          bubbleText = `⚠️ Authentication required: Please connect your Gmail account via Connect Gmail.`;
        } else if (msg.toLowerCase().includes("no refresh token") || msg.includes("500")) {
          errorBanner = `Request failed: ${msg}`;
          bubbleText = `⚠️ Error: ${msg}`;
        } else {
          errorBanner = `Request failed: ${msg}`;
          bubbleText = `⚠️ Error: ${msg}`;
        }
      }
      setError(errorBanner);
      // also push an assistant error bubble so chat history shows it
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: bubbleText,
          sources: [],
        },
      ]);
    } finally {
      setLoading(false);
      // focus input again
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 antialiased flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-20 backdrop-blur supports-[backdrop-filter]:bg-slate-950/70 bg-slate-950 border-b border-slate-800">
        <div className="mx-auto max-w-5xl px-4 sm:px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <div className="h-9 w-9 rounded-xl bg-indigo-600 flex items-center justify-center shrink-0 shadow-lg shadow-indigo-600/20">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div className="min-w-0">
              <h1 className="text-[15px] sm:text-lg font-semibold tracking-tight leading-none">AI Email Assistant</h1>
              <p className="text-xs text-slate-400 hidden sm:block">Ask about invoices, contests, and recent mail</p>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {/* Auth UI: Connect Gmail or user badge + Disconnect */}
            {!isAuthenticated ? (
              <button
                onClick={handleConnect}
                className="inline-flex items-center gap-2 rounded-full bg-indigo-600 px-4 py-2 text-xs font-semibold text-white hover:bg-indigo-500 active:bg-indigo-700 transition-colors shadow-lg shadow-indigo-600/20"
                aria-label="Connect Gmail"
              >
                <Mail className="h-3.5 w-3.5" />
                Connect Gmail
              </button>
            ) : (
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-2 rounded-full border border-emerald-900/50 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 max-w-[180px] truncate">
                  <span className="h-2 w-2 rounded-full bg-emerald-500 shrink-0" />
                  <Mail className="h-3 w-3 text-emerald-400 shrink-0" />
                  <span className="truncate">{userEmail}</span>
                </span>
                <button
                  onClick={handleDisconnect}
                  className="inline-flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
                  aria-label="Disconnect"
                >
                  <LogOut className="h-3.5 w-3.5" />
                  Disconnect
                </button>
              </div>
            )}

            {health === "checking" && (
              <span className="inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-400">
                <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
                Checking…
              </span>
            )}
            {health === "online" && (
              <span className="inline-flex items-center gap-2 rounded-full border border-emerald-900/50 bg-emerald-950/40 px-3 py-1.5 text-xs font-medium text-emerald-300">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
                <Wifi className="h-3.5 w-3.5" />
                Connected
              </span>
            )}
            {health === "offline" && (
              <span className="inline-flex items-center gap-2 rounded-full border border-red-900/50 bg-red-950/40 px-3 py-1.5 text-xs font-medium text-red-300">
                <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
                <WifiOff className="h-3.5 w-3.5" />
                Offline
              </span>
            )}
            {/* Sync Inbox button - next to health badge */}
            <button
              onClick={sendSync}
              disabled={isSyncing || health === "offline" || !isAuthenticated}
              className="inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-800 hover:border-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              aria-label="Sync Inbox"
              title={!isAuthenticated ? "Connect Gmail to sync" : "Sync Inbox"}
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isSyncing ? "animate-spin" : ""}`} />
              Sync Inbox
            </button>
            <button
              onClick={checkHealth}
              title="Retry health check"
              className="inline-flex items-center justify-center h-8 w-8 rounded-full bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-100 hover:border-slate-700 transition-colors"
              aria-label="Retry connection"
            >
              <RefreshCw className={`h-4 w-4 ${health === "checking" ? "animate-spin" : ""}`} />
            </button>
          </div>
        </div>
      </header>

      {/* Main chat area */}
      <main className="flex-1 mx-auto w-full max-w-3xl px-4 sm:px-6 py-6 sm:py-8 flex flex-col min-h-0">
        {/* Auth banner when not authenticated */}
        {!isAuthenticated && (
          <div className="mb-4 rounded-xl border border-amber-900/50 bg-amber-950/30 px-4 py-3 text-sm text-amber-200 flex items-center gap-3">
            <Mail className="h-4 w-4 text-amber-400 shrink-0" />
            <p className="flex-1 leading-relaxed">Connect your Gmail account to start querying emails</p>
          </div>
        )}
        {/* Sync success toast */}
        {syncMessage && (
          <div className="mb-4 rounded-xl border border-emerald-900/50 bg-emerald-950/30 px-4 py-3 text-sm text-emerald-200 flex items-start gap-3">
            <span className="mt-0.5 h-2 w-2 rounded-full bg-emerald-500 shrink-0 animate-pulse" />
            <p className="flex-1 leading-relaxed">{syncMessage}</p>
            <button
              onClick={() => setSyncMessage(null)}
              className="text-emerald-300 hover:text-white text-xs underline shrink-0"
            >
              Dismiss
            </button>
          </div>
        )}
        {/* Inline error bubble */}
        {error && (
          <div className="mb-4 rounded-xl border border-red-900/50 bg-red-950/30 px-4 py-3 text-sm text-red-200 flex items-start gap-3">
            <span className="mt-0.5 h-2 w-2 rounded-full bg-red-500 shrink-0 animate-pulse" />
            <p className="flex-1 leading-relaxed">{error}</p>
            <button
              onClick={() => setError(null)}
              className="text-red-300 hover:text-white text-xs underline shrink-0"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 space-y-4 overflow-y-auto pb-4 scroll-smooth">
          {messages.length === 0 ? (
            <div className="py-10 sm:py-16">
              <div className="mx-auto max-w-xl text-center">
                <div className="mx-auto h-14 w-14 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mb-4">
                  <Bot className="h-7 w-7 text-indigo-400" />
                </div>
                <h2 className="text-xl font-semibold tracking-tight">How can I help with your inbox?</h2>
                <p className="mt-2 text-sm text-slate-400 leading-relaxed">
                  Try one of these queries, or ask anything in natural language. I&apos;ll search your emails and cite the sources.
                </p>

                <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-3 text-left">
                  {PRESET_QUERIES.map(({ label, icon: Icon }) => (
                    <button
                      key={label}
                      onClick={() => sendQuery(label)}
                      disabled={!isAuthenticated}
                      className="group flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900 px-4 py-3 text-sm text-slate-200 hover:bg-slate-800/70 hover:border-slate-700 transition-colors text-left disabled:opacity-50 disabled:cursor-not-allowed"
                      title={!isAuthenticated ? "Connect Gmail to query" : label}
                    >
                      <span className="h-8 w-8 rounded-lg bg-slate-800 group-hover:bg-indigo-600 flex items-center justify-center shrink-0 transition-colors">
                        <Icon className="h-4 w-4 text-slate-300 group-hover:text-white" />
                      </span>
                      <span className="leading-snug">{label}</span>
                    </button>
                  ))}
                </div>

                <div className="mt-6 flex flex-wrap justify-center gap-2">
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-900 border border-slate-800 px-3 py-1 text-xs text-slate-400">
                    <MessageCircle className="h-3 w-3" /> Natural language
                  </span>
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-900 border border-slate-800 px-3 py-1 text-xs text-slate-400">
                    <Mail className="h-3 w-3" /> Cited emails
                  </span>
                </div>
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => {
              const isUser = msg.role === "user";
              const isExpanded = !!expandedSources[idx];
              return (
                <div key={`${msg.role}-${idx}-${msg.text.slice(0, 20)}`} className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
                  {!isUser && (
                    <div className="h-8 w-8 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center shrink-0 mt-1">
                      <Bot className="h-4 w-4 text-indigo-400" />
                    </div>
                  )}

                  <div className={`max-w-[85%] sm:max-w-[78%] ${isUser ? "order-first" : ""}`}>
                    <div
                      className={`rounded-2xl px-4 py-3 border shadow-sm leading-relaxed text-sm whitespace-pre-wrap break-words ${isUser
                          ? "bg-indigo-600 border-indigo-500 text-white rounded-br-md"
                          : "bg-slate-900 border-slate-800 text-slate-100 rounded-bl-md"
                        }`}
                    >
                      {msg.text ? (
                        <p>{unescapeHTML(msg.text)}</p>
                      ) : (
                        <p className="text-slate-400 italic">No response</p>
                      )}

                      {/* Collapsible Email Sources */}
                      {!isUser && msg.sources && msg.sources.length > 0 && (
                        <div className="mt-3">
                          <button
                            onClick={() => toggleSources(idx)}
                            className="inline-flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-800/50 hover:bg-slate-800 px-3 py-1.5 text-xs font-medium text-indigo-300 hover:text-indigo-200 transition-colors"
                          >
                            {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                            Referenced Emails
                            <span className="ml-1 rounded-full bg-indigo-600 text-white px-1.5 py-0.5 text-[10px] leading-none">
                              {msg.sources.length}
                            </span>
                          </button>

                          {isExpanded && (
                            <div className="mt-3 space-y-2.5">
                              {msg.sources.map((src, sIdx) => {
                                const sender = src.sender || src.from || src.from_address || src.author || "Unknown sender";
                                const subject = src.subject || src.title || "";
                                const date = src.date || src.timestamp || src.time || "";
                                const snippet = src.snippet || src.preview || src.body || src.text || src.content || "";
                                return (
                                  <div
                                    key={sIdx}
                                    className="rounded-xl border border-slate-800 bg-slate-800/40 overflow-hidden"
                                  >
                                    <div className="px-3.5 py-3">
                                      <div className="flex items-start justify-between gap-2">
                                        <p className="text-xs font-medium text-slate-200 leading-snug line-clamp-1">{unescapeHTML(subject || sender)}</p>
                                        {date && (
                                          <span className="text-[11px] text-slate-500 shrink-0">{formatDate(String(date))}</span>
                                        )}
                                      </div>
                                      {subject && sender && (
                                        <p className="text-[11px] text-slate-400 mt-0.5 line-clamp-1">{unescapeHTML(String(sender))}</p>
                                      )}
                                      {!subject && (
                                        <p className="text-xs text-slate-300 mt-0.5">{unescapeHTML(String(sender))}</p>
                                      )}
                                      {snippet && (
                                        <p className="mt-2 text-xs leading-relaxed text-slate-300 whitespace-pre-wrap break-words">
                                          {unescapeHTML(String(snippet))}
                                        </p>
                                      )}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* meta */}
                    <div className={`mt-1.5 flex items-center gap-2 text-[11px] text-slate-500 ${isUser ? "justify-end" : "justify-start"}`}>
                      {isUser ? (
                        <>
                          <span>You</span>
                          <User className="h-3 w-3" />
                        </>
                      ) : (
                        <>
                          <Bot className="h-3 w-3" />
                          <span>Assistant</span>
                        </>
                      )}
                    </div>
                  </div>

                  {isUser && (
                    <div className="h-8 w-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0 mt-1">
                      <User className="h-4 w-4 text-white" />
                    </div>
                  )}
                </div>
              );
            })
          )}

          {/* Loading dots */}
          {loading && (
            <div className="flex gap-3 justify-start">
              <div className="h-8 w-8 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center shrink-0">
                <Bot className="h-4 w-4 text-indigo-400" />
              </div>
              <div className="rounded-2xl rounded-bl-md bg-slate-900 border border-slate-800 px-4 py-3.5 flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-slate-600 animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="h-2 w-2 rounded-full bg-slate-600 animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="h-2 w-2 rounded-full bg-slate-600 animate-bounce" style={{ animationDelay: "300ms" }} />
                <span className="ml-2 text-xs text-slate-500">Thinking…</span>
              </div>
            </div>
          )}

          <div ref={endRef} />
        </div>

        {/* Composer */}
        <div className="sticky bottom-0 pt-4 bg-gradient-to-t from-slate-950 via-slate-950 to-transparent">
          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-2 sm:p-3 shadow-xl shadow-black/20">
            <div className="flex items-end gap-2">
              <div className="flex-1 relative">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={isAuthenticated ? "Ask about your emails… (e.g., When is my next contest?)" : "Connect Gmail to start querying emails"}
                  rows={1}
                  disabled={!isAuthenticated}
                  className="w-full resize-none max-h-28 rounded-xl bg-slate-800 border border-slate-700/50 px-4 py-3 pr-4 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-600 focus:border-indigo-600 transition-colors leading-relaxed disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ minHeight: "44px" }}
                  onInput={(e) => {
                    const t = e.target;
                    t.style.height = "auto";
                    t.style.height = Math.min(t.scrollHeight, 112) + "px";
                  }}
                />
              </div>
              <button
                onClick={() => sendQuery()}
                disabled={loading || !input.trim() || !isAuthenticated}
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 sm:px-5 py-3 text-sm font-medium text-white hover:bg-indigo-500 active:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shrink-0 shadow-lg shadow-indigo-600/20"
              >
                <span className="hidden sm:inline">{loading ? "Sending…" : "Send"}</span>
                <Send className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-2 flex items-center justify-between px-1">
              <p className="text-[11px] text-slate-500">Press Enter to send, Shift+Enter for new line</p>
              <p className="text-[11px] text-slate-600 hidden sm:block">Powered by your inbox</p>
            </div>
          </div>
        </div>
      </main>

      <footer className="py-4 text-center text-[11px] text-slate-600 border-t border-slate-900 mt-auto">
        AI Email Assistant • Connected to Gmail via FastAPI
      </footer>
    </div>
  );
}
