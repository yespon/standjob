import { ChatMessage } from "@/lib/api";

interface Props {
  message: ChatMessage;
}

export function ChatBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? "bg-blue-600 text-white rounded-br-md"
            : "bg-white text-gray-800 border border-gray-200 rounded-bl-md shadow-sm"
        }`}
      >
        {!isUser && (
          <div className="text-xs text-gray-400 mb-1 font-medium">
            🤖 辅导助手
          </div>
        )}
        <div>{message.content}</div>
      </div>
    </div>
  );
}
