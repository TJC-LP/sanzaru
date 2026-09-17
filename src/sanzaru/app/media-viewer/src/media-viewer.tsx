import type { App } from "@modelcontextprotocol/ext-apps";
import { useApp } from "@modelcontextprotocol/ext-apps/react";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { StrictMode, useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./global.css";

/** Shape of the tool input received via ontoolinput (only has tool args). */
interface MediaToolInput {
  filename: string;
  media_type: "video" | "audio" | "image";
  size_bytes?: number;
  mime_type?: string;
}

/** Shape returned by _get_media_data server tool. */
interface MediaDataChunk {
  data: string; // base64
  offset: number;
  chunk_size: number;
  total_size: number;
  is_last: boolean;
  mime_type: string;
}

const CHUNK_SIZE = 2 * 1024 * 1024; // 2 MB
const BASE64_DECODE_SLICE_SIZE = 256 * 1024; // Must stay divisible by 4

/** Deliberately generous: a host that opens a native "Save as..." dialog does not
 *  answer until the person has picked a folder, and the SDK's 60 s default would
 *  reject — and send notifications/cancelled — while that dialog is still open. */
const DOWNLOAD_TIMEOUT_MS = 10 * 60 * 1000;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function yieldToHost(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function decodeBase64Blob(data: string): Promise<Blob> {
  const parts: BlobPart[] = [];

  for (let start = 0; start < data.length; start += BASE64_DECODE_SLICE_SIZE) {
    const slice = data.slice(start, Math.min(start + BASE64_DECODE_SLICE_SIZE, data.length));
    const binary = atob(slice);
    const bytes = new Uint8Array(binary.length);

    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    parts.push(bytes);

    if (start + BASE64_DECODE_SLICE_SIZE < data.length) {
      await yieldToHost();
    }
  }

  return new Blob(parts);
}

/**
 * Base64-encode a Blob, letting the browser do the work.
 *
 * FileReader encodes off the main thread and hands back one string, so unlike a
 * hand-rolled btoa() loop there is no frame to block, no ArrayBuffer copy and no
 * array of partial results to hold alongside the finished one — which matters at
 * the size of a rendered episode. The read is not a fetch, so a strict MCP App
 * `connect-src` cannot block it the way it can block fetch("data:...").
 */
async function encodeBlobBase64(blob: Blob): Promise<string> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the media for download"));
    reader.readAsDataURL(blob);
  });

  // "data:<mime>;base64,<payload>" — everything up to the first comma is the header.
  return dataUrl.slice(dataUrl.indexOf(",") + 1);
}

/** Extract parsed JSON from a CallToolResult — tries structuredContent first, then text content. */
function extractJson<T>(result: CallToolResult): T | null {
  // Try structuredContent first (available when tool has structured output)
  if (result.structuredContent) {
    return result.structuredContent as unknown as T;
  }
  // Fall back to parsing JSON from text content blocks
  const textBlock = result.content?.find((c) => c.type === "text");
  if (textBlock && "text" in textBlock) {
    try {
      return JSON.parse(textBlock.text) as T;
    } catch {
      return null;
    }
  }
  return null;
}

function MediaViewer() {
  const [mediaInput, setMediaInput] = useState<MediaToolInput | null>(null);
  const [appError, setAppError] = useState<string | null>(null);

  const { app, error } = useApp({
    appInfo: { name: "Sanzaru Media Viewer", version: "1.0.0" },
    capabilities: {},
    onAppCreated: (app) => {
      // ontoolinput fires first with the tool's input arguments (media_type + filename).
      // size_bytes and mime_type are not available yet — the MediaPlayer handles that.
      app.ontoolinput = async (input) => {
        const args = input.arguments as unknown as MediaToolInput;
        if (args?.filename && args?.media_type) {
          setMediaInput(args);
        }
      };
      // ontoolresult fires after the tool completes with the full result (includes size_bytes, mime_type).
      // If ontoolinput didn't fire (e.g., some hosts skip it), this is the fallback.
      app.ontoolresult = async (result) => {
        const parsed = extractJson<MediaToolInput>(result);
        if (parsed?.filename && parsed?.media_type) {
          setMediaInput((prev) => prev ?? parsed);
        }
      };
      app.onerror = (err) => {
        console.error(err);
        setAppError(err instanceof Error ? err.message : String(err));
      };
    },
  });

  if (error) return <div className="media-viewer"><span className="error">Error: {error.message}</span></div>;
  if (appError) return <div className="media-viewer"><span className="error">Error: {appError}</span></div>;
  if (!app) return <div className="media-viewer"><span className="status">Connecting...</span></div>;
  if (!mediaInput) return <div className="media-viewer"><span className="status">Waiting for media...</span></div>;

  return <MediaPlayer app={app} input={mediaInput} />;
}

interface MediaPlayerProps {
  app: App;
  input: MediaToolInput;
}

function MediaPlayer({ app, input }: MediaPlayerProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [totalSize, setTotalSize] = useState<number | null>(input.size_bytes ?? null);
  const [mimeType, setMimeType] = useState<string | null>(input.mime_type ?? null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<{ text: string; kind: "status" | "error" } | null>(null);
  const blobUrlRef = useRef<string | null>(null);
  const blobRef = useRef<Blob | null>(null);

  // `ui/download-file` is not in the stable 2026-01-26 Apps spec, so a host may
  // not implement it. Without the control the file is still playable — it just
  // cannot be saved from here — so this hides rather than degrades.
  const canDownload = app.getHostCapabilities()?.downloadFile != null;

  const loadMedia = useCallback(async () => {
    setLoading(true);
    setErrorMsg(null);
    setProgress(0);

    setSaveMessage(null);

    // Revoke previous blob URL
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
    blobRef.current = null;

    try {
      const chunks: Blob[] = [];
      let offset = 0;
      let done = false;
      let resolvedMime: string | null = input.mime_type ?? null;

      while (!done) {
        const result = await app.callServerTool({
          name: "_get_media_data",
          arguments: {
            media_type: input.media_type,
            filename: input.filename,
            offset,
            chunk_size: CHUNK_SIZE,
          },
        });

        const chunk = extractJson<MediaDataChunk>(result);
        if (!chunk?.data) {
          throw new Error("No data received from server");
        }

        // On first chunk, capture total_size and mime_type from the server
        if (offset === 0) {
          setTotalSize(chunk.total_size);
          setMimeType(chunk.mime_type);
          resolvedMime = chunk.mime_type;
        }

        // Decode in small slices so host timers and message handlers get
        // turns during large media loads. Using fetch(data:...) would also
        // yield, but strict MCP App CSP hosts can block it via connect-src.
        const chunkBlob = await decodeBase64Blob(chunk.data);
        chunks.push(chunkBlob);

        offset += chunk.chunk_size;
        done = chunk.is_last;

        // Update progress
        if (chunk.total_size > 0) {
          setProgress(Math.min(100, Math.round((offset / chunk.total_size) * 100)));
        }
      }

      // Assemble blob using server-reported mime type
      const blob = new Blob(chunks, { type: resolvedMime ?? "application/octet-stream" });
      const url = URL.createObjectURL(blob);
      blobUrlRef.current = url;
      blobRef.current = blob;
      setBlobUrl(url);
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [app, input.filename, input.media_type]);

  const downloadMedia = useCallback(async () => {
    const blob = blobRef.current;
    if (!blob) return;

    setSaving(true);
    setSaveMessage(null);
    try {
      // The bytes are already here — assembled for playback — so saving costs one
      // encode rather than a second trip to the server. Handing them over as an
      // embedded resource also means no credential and no URL is involved, which
      // is what makes this work identically over stdio and HTTP.
      const base64 = await encodeBlobBase64(blob);

      // Defensive: the server only ever reports a bare basename, and the host
      // derives the saved filename from the last path segment of this URI.
      const name = input.filename.split(/[\\/]/).pop() || "download";

      const { isError } = await app.downloadFile(
        {
          contents: [
            {
              type: "resource",
              resource: {
                // Encoded because this is a URI, not a path: nothing upstream
                // restricts which characters a filename may contain, and a raw
                // "#" or "?" would end the path component early for any host
                // that parses this rather than splitting it — saving the file
                // under a truncated name with no extension.
                uri: `file:///${encodeURIComponent(name)}`,
                mimeType: mimeType ?? "application/octet-stream",
                blob: base64,
              },
            },
          ],
        },
        { timeout: DOWNLOAD_TIMEOUT_MS },
      );

      // Usually the person dismissing the host's own save dialog, but the host
      // may also have refused outright, so this does not claim to know which.
      if (isError) {
        setSaveMessage({ text: "Download cancelled or refused", kind: "status" });
      }
    } catch (e) {
      setSaveMessage({ text: e instanceof Error ? e.message : String(e), kind: "error" });
    } finally {
      setSaving(false);
    }
  }, [app, input.filename, mimeType]);

  useEffect(() => {
    loadMedia();
    return () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
      }
    };
  }, [loadMedia]);

  return (
    <div className="media-viewer">
      <div className="file-row">
        <span className="filename">{input.filename}</span>
        {blobUrl && canDownload && (
          <button type="button" className="download" onClick={downloadMedia} disabled={saving}>
            {saving ? "Preparing..." : "Download"}
          </button>
        )}
      </div>
      <span className={`save-message ${saveMessage?.kind ?? "status"}`} aria-live="polite">
        {saveMessage?.text ?? ""}
      </span>

      {loading && (
        <div className="progress-container">
          <div className="progress-bar">
            <div className="fill" style={{ width: `${progress}%` }} />
          </div>
          <span className="progress-text">
            {totalSize != null
              ? `Loading... ${progress}% (${formatBytes(Math.round(totalSize * progress / 100))} / ${formatBytes(totalSize)})`
              : `Loading... ${progress}%`
            }
          </span>
        </div>
      )}

      {errorMsg && <span className="error">{errorMsg}</span>}

      {blobUrl && (
        <div className="media-container">
          {input.media_type === "video" && (
            <video src={blobUrl} controls />
          )}
          {input.media_type === "audio" && (
            <audio src={blobUrl} controls />
          )}
          {input.media_type === "image" && (
            <img src={blobUrl} alt={input.filename} decoding="async" />
          )}
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MediaViewer />
  </StrictMode>,
);
