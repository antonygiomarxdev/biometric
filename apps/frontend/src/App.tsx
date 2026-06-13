import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard from "@/pages/Dashboard";
import ComparisonView from "@/pages/ComparisonView";
import ScannerPage from "@/pages/ScannerPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/scanner" element={<ScannerPage />} />
        <Route path="/cases/:caseId/compare" element={<ComparisonView />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
