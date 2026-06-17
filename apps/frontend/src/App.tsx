import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard from "@/pages/Dashboard";
import ComparisonView from "@/pages/ComparisonView";
import EnrollPage from "@/pages/EnrollPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/enroll" element={<EnrollPage />} />
        <Route path="/cases/:caseId/enroll" element={<EnrollPage />} />
        <Route path="/cases/:caseId/compare" element={<ComparisonView />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
