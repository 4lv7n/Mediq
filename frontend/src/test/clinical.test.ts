import { describe, expect, it } from 'vitest';
import { featureGroups, getPrototypeResult, buildReport } from '../lib/clinical';
describe('MEDIQ research constraints', () => {
 it('contains exactly the requested 14 clinical features', () => { expect(featureGroups.flatMap(g => g.fields.map(f => f.label))).toEqual(['Age','Gender','ESR','CRP','RF','Anti-CCP','HLA-B27','ANA','Anti-Ro','Anti-La','Anti-dsDNA','Anti-Sm','C3','C4']); });
 it('never fabricates predictions for the frontend prototype', () => { expect(getPrototypeResult()).toEqual({ predictedClass: null, topClassScore: null, probabilities: null, contributions: null, status: 'not-connected' }); });
 it('exports supplied inputs without inventing missing patient values', () => { const report = buildReport({ age: '35', crp: '12.5' }); expect(report).toContain('Age: 35 years'); expect(report).toContain('CRP: 12.5 mg/L'); expect(report).toContain('Gender: Not supplied'); });
});
