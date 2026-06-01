"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  ChatMessage,
  startSessionStream,
  sendMessageStream,
  uploadFile,
  SessionExpiredError,
} from "@/lib/api";
import { ChatBubble } from "@/components/ChatBubble";
import { ChatInput } from "@/components/ChatInput";
import { StatusBar } from "@/components/StatusBar";

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [phase, setPhase] = useState("init");
  const [activeMode, setActiveMode] = useState("proactive");
  const [nextNodes, setNextNodes] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // 启动会话（流式）
  const handleStart = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await startSessionStream({
        onSession: (id) => setThreadId(id),
        onMessage: (msg) => setMessages((prev) => [...prev, msg]),
        onDone: (p, nodes, mode) => {
          setPhase(p);
          setNextNodes(nodes);
          setActiveMode(mode);
        },
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "启动失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    handleStart();
  }, [handleStart]);

  // 会话过期时自动重新初始化
  const restartSession = useCallback(async () => {
    setMessages([]);
    setThreadId(null);
    setPhase("init");
    setActiveMode("proactive");
    setNextNodes([]);
    setError("会话已过期，正在重新连接...");
    try {
      await startSessionStream({
        onSession: (id) => setThreadId(id),
        onMessage: (msg) => setMessages((prev) => [...prev, msg]),
        onDone: (p, nodes, mode) => {
          setPhase(p);
          setNextNodes(nodes);
          setActiveMode(mode);
        },
      });
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "重连失败，请刷新页面");
    }
  }, []);

  // 发送文本消息（流式）
  const handleSend = async (text: string) => {
    if (!threadId || loading) return;

    const userMsg: ChatMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setError(null);

    try {
      await sendMessageStream(threadId, text, undefined, {
        onMessage: (msg) => setMessages((prev) => [...prev, msg]),
        onDone: (p, nodes, mode) => {
          setPhase(p);
          setNextNodes(nodes);
          setActiveMode(mode);
        },
        onError: (detail) => setError(detail),
      });
    } catch (e) {
      if (e instanceof SessionExpiredError) {
        await restartSession();
      } else {
        setError(e instanceof Error ? e.message : "发送失败");
      }
    } finally {
      setLoading(false);
    }
  };

  // 上传文件并发送（流式）
  const handleFileUpload = async (file: File) => {
    if (!threadId || loading) return;

    setLoading(true);
    setError(null);

    try {
      const uploadRes = await uploadFile(file);
      const userMsg: ChatMessage = {
        role: "user",
        content: `📎 已上传文件：${uploadRes.filename}`,
      };
      const successMsg: ChatMessage = {
        role: "assistant",
        content: `✅ 文件上传成功！正在解析和评审中，请稍候...`,
      };
      setMessages((prev) => [...prev, userMsg, successMsg]);

      await sendMessageStream(
        threadId,
        `我已上传文件：${uploadRes.filename}`,
        uploadRes.file_path,
        {
          onMessage: (msg) => setMessages((prev) => [...prev, msg]),
          onDone: (p, nodes, mode) => {
            setPhase(p);
            setNextNodes(nodes);
            setActiveMode(mode);
          },
          onError: (detail) => setError(detail),
        }
      );
    } catch (e) {
      if (e instanceof SessionExpiredError) {
        await restartSession();
      } else {
        setError(e instanceof Error ? e.message : "上传失败");
      }
    } finally {
      setLoading(false);
    }
  };

  const waitingForFile = nextNodes.includes("wait_for_file");
  const waitingForReply = nextNodes.includes("wait_for_reply");
  const isDone = phase === "done";

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex-shrink-0">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-gray-900">
              📋 岗标 AI 教练 Demo
            </h1>
            <p className="text-sm text-gray-500 mt-0.5">
              基于 AI 的智能辅导
            </p>
          </div>
          <StatusBar phase={phase} loading={loading} mode={activeMode} />
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.map((msg, i) => (
            <ChatBubble key={i} message={msg} />
          ))}

          {loading && (
            <div className="flex items-center gap-2 text-gray-400 text-sm py-2">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
              </div>
              AI 正在思考...
            </div>
          )}

          {error && (
            <div className="bg-red-50 text-red-600 text-sm px-4 py-3 rounded-lg border border-red-200">
              {error}
            </div>
          )}

          {isDone && messages.length > 0 && (
            <div className="text-center text-gray-500 text-sm py-4">
              主动引导已完成，可继续提问进入被动答疑
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input */}
      <footer className="bg-white border-t border-gray-200 px-6 py-4 flex-shrink-0">
        <div className="max-w-3xl mx-auto">
          <ChatInput
            onSend={handleSend}
            onFileUpload={handleFileUpload}
            disabled={loading}
            showFileUpload={waitingForFile}
            placeholder={
              waitingForFile
                ? "上传 .xlsx 文件或直接发送消息..."
                : waitingForReply
                ? "请输入您的回答..."
                : isDone
                ? "主动引导已完成，可继续提问..."
                : "输入消息..."
            }
          />
        </div>
      </footer>
    </div>
  );
}
