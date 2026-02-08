import type { App } from "@modelcontextprotocol/ext-apps";
import { useApp } from "@modelcontextprotocol/ext-apps/react";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { StrictMode, useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./global.css";

/** Shape of the tool input received via ontoolinput / ontoolresult. */
interface MediaToolInput {
  filename: string;
  media_type: "video" | "audio" | "image";
  size_bytes: number;
  mime_type: string;
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

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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

  const { app, error } = useApp({
    appInfo: { name: "Sanzaru Media Viewer", version: "1.0.0" },
    capabilities: {},
    onAppCreated: (app) => {
      app.ontoolinput = async (input) => {
        const args = input.arguments as unknown as MediaToolInput;
        setMediaInput(args);
      };
      app.ontoolresult = async (result) => {
        const parsed = extractJson<MediaToolInput>(result);
        if (parsed?.filename) {
          setMediaInput(parsed);
        }
      };
      app.onerror = console.error;
    },
  });

  if (error) return <div className="media-viewer"><span className="error">Error: {error.message}</span></div>;
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
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const blobUrlRef = useRef<string | null>(null);

  const loadMedia = useCallback(async () => {
    setLoading(true);
    setErrorMsg(null);
    setProgress(0);

    // Revoke previous blob URL
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }

    try {
      const chunks: Uint8Array[] = [];
      let offset = 0;
      let done = false;

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

        // Decode base64 chunk
        const binary = atob(chunk.data);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        chunks.push(bytes);

        offset += chunk.chunk_size;
        done = chunk.is_last;

        // Update progress
        if (chunk.total_size > 0) {
          setProgress(Math.min(100, Math.round((offset / chunk.total_size) * 100)));
        }
      }

      // Assemble blob
      const blob = new Blob(chunks, { type: input.mime_type });
      const url = URL.createObjectURL(blob);
      blobUrlRef.current = url;
      setBlobUrl(url);
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [app, input.filename, input.media_type, input.mime_type]);

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
      <span className="filename">{input.filename}</span>

      {loading && (
        <div className="progress-container">
          <div className="progress-bar">
            <div className="fill" style={{ width: `${progress}%` }} />
          </div>
          <span className="progress-text">
            Loading... {progress}% ({formatBytes(Math.round(input.size_bytes * progress / 100))} / {formatBytes(input.size_bytes)})
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
            <img src={blobUrl} alt={input.filename} />
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
