import React from 'react';
import { AlertCircle, ShieldAlert } from 'lucide-react';

export const SafetyGuidance: React.FC = () => {
  return (
    <div className="bg-amber-50 border-l-4 border-amber-500 p-4 my-6 rounded-r-md">
      <div className="flex items-center mb-2">
        <ShieldAlert className="text-amber-600 mr-2" size={20} />
        <h3 className="text-lg font-semibold text-amber-900">Safety & Behavior Guidance</h3>
      </div>
      <div className="text-sm text-amber-800 space-y-2">
        <p>
          <strong>Frontend-Only Analysis Guardrails:</strong> To avoid fabricated or misleading outputs, the application should adhere to the following:
        </p>
        <ul className="list-disc ml-5 space-y-1">
          <li><strong>No Generative Interpretation:</strong> Use static, curated reference data. Do not use LLMs to "interpret" symptoms or lab results.</li>
          <li><strong>Zero Diagnostic Claims:</strong> Explicitly state that information is for educational purposes only. Never provide "probability" or "confidence" scores for a disease.</li>
          <li><strong>Input Boundary Checks:</strong> If a user enters data, it should only be matched against known clinical reference ranges. Unexpected inputs must be flagged for professional review.</li>
          <li><strong>Contextual Disclaimer:</strong> Every summary must be accompanied by a reminder that diagnosis requires comprehensive clinical evaluation by a specialist.</li>
          <li><strong>Data Integrity:</strong> Do not synthesize new disease categories or symptom combinations.</li>
        </ul>
      </div>
    </div>
  );
};
