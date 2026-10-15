import React from 'react';
import { DISEASE_SUMMARIES } from '../../lib/disease-data';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Stethoscope, Activity, UserCog, ClipboardCheck, AlertTriangle } from 'lucide-react';

export const DiseaseReference: React.FC = () => {
  return (
    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-1">
      {DISEASE_SUMMARIES.map((disease) => (
        <Card key={disease.id} className="overflow-hidden border-slate-200">
          <CardHeader className="bg-slate-50 border-b border-slate-100">
            <CardTitle className="text-xl text-slate-800">{disease.name}</CardTitle>
          </CardHeader>
          <CardContent className="pt-4 grid gap-4 text-sm">
            <div className="flex gap-3">
              <Stethoscope className="text-blue-500 shrink-0" size={18} />
              <div>
                <span className="font-semibold text-slate-700">Symptoms:</span> {disease.symptoms}
              </div>
            </div>
            <div className="flex gap-3">
              <Activity className="text-emerald-500 shrink-0" size={18} />
              <div>
                <span className="font-semibold text-slate-700">Biomarkers:</span> {disease.biomarkers}
              </div>
            </div>
            <div className="flex gap-3">
              <UserCog className="text-purple-500 shrink-0" size={18} />
              <div>
                <span className="font-semibold text-slate-700">Specialist:</span> {disease.specialist}
              </div>
            </div>
            <div className="flex gap-3">
              <ClipboardCheck className="text-indigo-500 shrink-0" size={18} />
              <div>
                <span className="font-semibold text-slate-700">Tests:</span> {disease.tests}
              </div>
            </div>
            <div className="flex gap-3">
              <AlertTriangle className="text-rose-500 shrink-0" size={18} />
              <div>
                <span className="font-semibold text-slate-700">Complications:</span> {disease.complications}
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};
