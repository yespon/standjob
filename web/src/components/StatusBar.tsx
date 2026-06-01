interface Props {
  phase: string;
  loading: boolean;
  mode?: string;
}

const PHASE_LABELS: Record<string, { label: string; color: string }> = {
  init: { label: "初始化", color: "bg-gray-100 text-gray-600" },
  loaded: { label: "等待上传", color: "bg-yellow-100 text-yellow-700" },
  reviewing: { label: "评审中", color: "bg-blue-100 text-blue-700" },
  guiding: { label: "辅导中", color: "bg-green-100 text-green-700" },
  done: { label: "已完成", color: "bg-gray-100 text-gray-500" },
};

const MODE_LABELS: Record<string, { label: string; color: string }> = {
  proactive: { label: "主动引导", color: "bg-emerald-100 text-emerald-700" },
  reactive_qa: { label: "被动答疑", color: "bg-orange-100 text-orange-700" },
};

export function StatusBar({ phase, loading, mode = "proactive" }: Props) {
  const info = PHASE_LABELS[phase] || PHASE_LABELS.init;
  const modeInfo = MODE_LABELS[mode] || MODE_LABELS.proactive;

  return (
    <div className="flex items-center gap-2">
      {loading && (
        <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      )}
      <span
        className={`text-xs font-medium px-2.5 py-1 rounded-full ${info.color}`}
      >
        {info.label}
      </span>
      <span
        className={`text-xs font-medium px-2.5 py-1 rounded-full ${modeInfo.color}`}
      >
        {modeInfo.label}
      </span>
    </div>
  );
}
