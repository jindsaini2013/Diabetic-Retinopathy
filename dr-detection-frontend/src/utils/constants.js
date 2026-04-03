export const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const DR_GRADES = {
  0: { label: "No DR",          color: "text-emerald-400", bg: "bg-emerald-400/10", border: "border-emerald-400/30", dot: "bg-emerald-400" },
  1: { label: "Mild DR",        color: "text-yellow-400",  bg: "bg-yellow-400/10",  border: "border-yellow-400/30",  dot: "bg-yellow-400"  },
  2: { label: "Moderate DR",    color: "text-orange-400",  bg: "bg-orange-400/10",  border: "border-orange-400/30",  dot: "bg-orange-400"  },
  3: { label: "Severe DR",      color: "text-red-400",     bg: "bg-red-400/10",     border: "border-red-400/30",     dot: "bg-red-400"     },
};

export const MAX_HISTORY = 5;
