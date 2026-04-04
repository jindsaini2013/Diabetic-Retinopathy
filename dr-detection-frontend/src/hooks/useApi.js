import { useState, useCallback, useEffect } from "react";
import axios from "axios";
import { API_BASE_URL, MAX_HISTORY } from "../utils/constants";

export function useApi() {
  const [health, setHealth]           = useState("checking");
  const [prediction, setPrediction]   = useState(null);
  const [heatmap, setHeatmap]         = useState(null);
  const [history, setHistory]         = useState([]);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingHeatmap, setLoadingHeatmap] = useState(false);
  const [error, setError]             = useState(null);

  // Poll health every 15s
  const checkHealth = useCallback(async () => {
    try {
      await axios.get(`${API_BASE_URL}/health`, { timeout: 30000 });
      setHealth("online");
    } catch {
      setHealth((current) => (current === "online" ? "waking" : "offline"));
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const id = setInterval(checkHealth, 15000);
    return () => clearInterval(id);
  }, [checkHealth]);

  const predict = useCallback(async (file) => {
    setError(null);
    setHeatmap(null);
    setPrediction(null);
    setLoadingPredict(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await axios.post(`${API_BASE_URL}/predict`, form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 30000,
      });
      const result = { ...data, timestamp: Date.now(), filename: file.name };
      setPrediction(result);
      setHistory((prev) => [result, ...prev].slice(0, MAX_HISTORY));
      return result;
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Prediction failed.";
      setError(msg);
      return null;
    } finally {
      setLoadingPredict(false);
    }
  }, []);

  const fetchHeatmap = useCallback(async (file) => {
    setError(null);
    setLoadingHeatmap(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await axios.post(`${API_BASE_URL}/gradcam`, form, {
        headers: { "Content-Type": "multipart/form-data" },
        responseType: "blob",
        timeout: 30000,
      });
      const url = URL.createObjectURL(data);
      setHeatmap(url);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Grad-CAM failed.";
      setError(msg);
    } finally {
      setLoadingHeatmap(false);
    }
  }, []);

  const clear = useCallback(() => {
    setPrediction(null);
    setHeatmap(null);
    setError(null);
  }, []);

  return {
    health, prediction, heatmap, history,
    loadingPredict, loadingHeatmap, error,
    predict, fetchHeatmap, clear,
  };
}
