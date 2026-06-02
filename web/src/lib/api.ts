const API_BASE = "";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface StartSessionResponse {
  thread_id: string;
  messages: ChatMessage[];
  phase: string;
  next_nodes: string[];
  active_mode: string;
  current_focus_id?: string | null;
  stuck_counter?: number;
  hint_level?: number;
  closure_summary?: string | null;
}

export interface ChatResponse {
  messages: ChatMessage[];
  phase: string;
  next_nodes: string[];
  active_mode: string;
  current_focus_id?: string | null;
  stuck_counter?: number;
  hint_level?: number;
  closure_summary?: string | null;
}

export interface SessionState {
  phase: string;
  next_nodes: string[];
  current_item_index: number;
  total_items: number;
  reflection_round: number;
  active_mode: string;
  current_focus_id?: string | null;
  stuck_counter?: number;
  hint_level?: number;
  closure_summary?: string | null;
  rubric_eval_summary?: {
    checked_item_ids?: number[];
    matched_item_ids?: number[];
    relaxed_item_ids?: number[];
    coverage_ok?: boolean;
  };
}

export interface UploadResponse {
  file_path: string;
  filename: string;
}

export interface SSECallbacks {
  onSession?: (threadId: string) => void;
  onMessage: (msg: ChatMessage) => void;
  onDone: (state: SessionState) => void;
  onError?: (error: string) => void;
}

async function consumeSSE(response: Response, callbacks: SSECallbacks) {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("无法读取流");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let eventType = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const data = JSON.parse(line.slice(6));
        switch (eventType) {
          case "session":
            callbacks.onSession?.(data.thread_id);
            break;
          case "message":
            callbacks.onMessage(data as ChatMessage);
            break;
          case "error":
            callbacks.onError?.(data.detail || "未知错误");
            break;
          case "done":
            callbacks.onDone({
              phase: data.phase,
              next_nodes: data.next_nodes || [],
              current_item_index: data.current_item_index || 0,
              total_items: data.total_items || 0,
              reflection_round: data.reflection_round || 0,
              active_mode: data.active_mode || "proactive",
              current_focus_id: data.current_focus_id,
              stuck_counter: data.stuck_counter || 0,
              hint_level: data.hint_level || 0,
              closure_summary: data.closure_summary || null,
              rubric_eval_summary: data.rubric_eval_summary || {},
            });
            break;
        }
        eventType = "";
      }
    }
  }
}

export async function startSessionStream(callbacks: SSECallbacks) {
  const res = await fetch(`${API_BASE}/api/session/start/stream`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`启动会话失败: ${res.statusText}`);
  await consumeSSE(res, callbacks);
}

export class SessionExpiredError extends Error {
  constructor() {
    super("会话已过期，正在重新连接...");
    this.name = "SessionExpiredError";
  }
}

export async function sendMessageStream(
  threadId: string,
  message: string,
  filePath: string | undefined,
  callbacks: SSECallbacks
) {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      thread_id: threadId,
      message,
      file_path: filePath || null,
    }),
  });
  if (res.status === 404) throw new SessionExpiredError();
  if (!res.ok) throw new Error(`发送消息失败: ${res.statusText}`);
  await consumeSSE(res, callbacks);
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`上传文件失败: ${res.statusText}`);
  return res.json();
}
