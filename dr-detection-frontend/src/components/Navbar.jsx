import { motion } from "framer-motion";
import { Eye, Activity } from "lucide-react";

export default function Navbar({ health }) {
  const isChecking = health === "checking";
  const isOnline = health === "online";
  const isWaking = health === "waking";

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="fixed top-0 left-0 right-0 z-50 glass-dark border-b border-slate-700/40"
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-cyan-500 flex items-center justify-center shadow-glow">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="font-display font-bold text-lg text-white tracking-tight">
              Retina<span className="text-gradient">AI</span>
            </span>
            <p className="text-[10px] text-slate-500 leading-none font-mono uppercase tracking-widest">
              DR Detection System
            </p>
          </div>
        </div>

        {/* Health Status */}
        <div className="flex items-center gap-2 glass rounded-full px-4 py-2 text-sm">
          <Activity className="w-4 h-4 text-slate-400" />
          <span className="text-slate-400 font-mono text-xs">API</span>
          <span
            className={`w-2 h-2 rounded-full ${
              isChecking
                ? "bg-slate-500 animate-pulse"
                : isOnline
                ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]"
                : isWaking
                ? "bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.8)] animate-pulse"
                : "bg-red-400 shadow-[0_0_6px_rgba(248,113,113,0.8)]"
            }`}
          />
          <span
            className={`text-xs font-mono font-medium ${
              isChecking
                ? "text-slate-500"
                : isOnline
                ? "text-emerald-400"
                : isWaking
                ? "text-amber-400"
                : "text-red-400"
            }`}
          >
            {isChecking ? "Checking…" : isOnline ? "Online" : isWaking ? "Waking up…" : "Offline"}
          </span>
        </div>
      </div>
    </motion.nav>
  );
}
