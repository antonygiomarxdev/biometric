import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { ToastProvider } from "./components/ui/toast";
import { QueryClientProvider } from "./lib/query";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider>
      <ToastProvider>
        <App />
      </ToastProvider>
    </QueryClientProvider>
  </StrictMode>,
);
