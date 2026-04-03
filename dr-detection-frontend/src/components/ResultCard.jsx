import { motion } from "framer-motion";
import { CheckCircle, TrendingUp, Clock, Scan } from "lucide-react";
import { DR_GRADES } from "../utils/constants";
import { formatConfidence, formatTime } from "../utils/helpers";

export default function ResultCard({ prediction, onFetchHeatmap, loadingHeatmap, heatmap }) {
  if (!prediction) return null;
  const grade = DR_GRADES[prediction.grade ?? prediction.class ?? 0];
  const confidence = prediction.confidence ?? prediction.score ?? 0;
  const classProbabilities = Object.entries(prediction.class_probabilities ?? {});

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="glass-dark rounded-2xl p-6 border border-slate-700/50 space-y-5"
    >
      {/* Header */}
      <div className="flex items-center gap-2 text-slate-400 text-xs font-mono uppercase tracking-widest">
        <CheckCircle className="w-4 h-4 text-brand-400" />
        Analysis Complete
      </div>

      {/* Grade */}
      <div className={`inline-flex items-center gap-3 px-5 py-3 rounded-xl border ${grade.bg} ${grade.border}`}>
        <span className={`w-2.5 h-2.5 rounded-full ${grade.dot} shadow-[0_0_8px_currentColor]`} />
        <span className={`font-display text-xl font-bold ${grade.color}`}>{grade.label}</span>
      </div>

      {/* Confidence bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-slate-400 flex items-center gap-1.5">
            <TrendingUp className="w-3.5 h-3.5" /> Confidence
          </span>
          <span className="font-mono font-semibold text-white">{formatConfidence(confidence)}</span>
        </div>
        <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
          <motion.div
            className={`h-full rounded-full ${grade.dot}`}
            initial={{ width: 0 }}
            animate={{ width: `${confidence * 100}%` }}
            transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
          />
        </div>
      </div>

      {/* Metadata */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-slate-900/60 rounded-xl p-3 border border-slate-800">
          <p className="text-slate-600 text-xs font-mono mb-1 flex items-center gap-1">
            <Scan className="w-3 h-3" /> File
          </p>
          <p className="text-slate-300 text-xs font-mono truncate">{prediction.filename}</p>
        </div>
        <div className="bg-slate-900/60 rounded-xl p-3 border border-slate-800">
          <p className="text-slate-600 text-xs font-mono mb-1 flex items-center gap-1">
            <Clock className="w-3 h-3" /> Time
          </p>
          <p className="text-slate-300 text-xs font-mono">{formatTime(prediction.timestamp)}</p>
        </div>
      </div>

      {classProbabilities.length > 0 && (
        <div className="space-y-3">
          <div className="text-slate-400 text-xs font-mono uppercase tracking-widest">
            All Class Probabilities
          </div>
          <div className="space-y-2">
            {classProbabilities.map(([key, entry]) => {
              const classStyle = DR_GRADES[key] ?? DR_GRADES[0];
              const percentage = entry.percentage ?? ((entry.probability ?? 0) * 100);

              return (
                <div key={key} className="bg-slate-900/60 rounded-xl p-3 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between gap-3 text-sm">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className={`w-2.5 h-2.5 rounded-full ${classStyle.dot}`} />
                      <span className="text-slate-200 truncate">{entry.label}</span>
                    </div>
                    <span className="font-mono text-white">{percentage.toFixed(2)}%</span>
                  </div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div
                      className={`h-full rounded-full ${classStyle.dot}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${percentage}%` }}
                      transition={{ duration: 0.7, ease: "easeOut" }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Grad-CAM button */}
      <button
        onClick={onFetchHeatmap}
        disabled={loadingHeatmap || !!heatmap}
        className={`w-full py-3 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 border transition-all duration-300
          ${heatmap
            ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400 cursor-default"
            : loadingHeatmap
            ? "bg-slate-800 border-slate-700 text-slate-500 cursor-not-allowed"
            : "bg-brand-500/10 border-brand-500/30 text-brand-400 hover:bg-brand-500/20 hover:border-brand-400/50 active:scale-[0.98]"
          }`}
      >
        {heatmap ? "✓ Heatmap Loaded" : loadingHeatmap ? "Generating Heatmap…" : "⬡ Generate Grad-CAM Heatmap"}
      </button>
    </motion.div>
  );
}
