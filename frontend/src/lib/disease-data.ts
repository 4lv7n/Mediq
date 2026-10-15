export interface DiseaseSummary {
  id: string;
  name: string;
  symptoms: string;
  biomarkers: string;
  specialist: string;
  tests: string;
  complications: string;
}

export const DISEASE_SUMMARIES: DiseaseSummary[] = [
  {
    id: "ra",
    name: "Rheumatoid Arthritis (RA)",
    symptoms: "Symmetrical joint pain and swelling (often hands and feet), morning stiffness lasting over 30 minutes, fatigue.",
    biomarkers: "Rheumatoid Factor (RF), Anti-Cyclic Citrullinated Peptide (Anti-CCP).",
    specialist: "Rheumatologist",
    tests: "Inflammatory markers (ESR, CRP), Joint Ultrasound, X-ray of hands/feet.",
    complications: "Joint deformity, interstitial lung disease, increased cardiovascular risk."
  },
  {
    id: "sle",
    name: "Systemic Lupus Erythematosus (SLE)",
    symptoms: "Malar (butterfly) rash, photosensitivity, joint pain, fatigue, mouth ulcers.",
    biomarkers: "ANA (screening), Anti-dsDNA, Anti-Smith.",
    specialist: "Rheumatologist",
    tests: "Complete Blood Count (CBC), Urinalysis (for protein), Complement levels (C3, C4).",
    complications: "Lupus nephritis, pleurisy, pericarditis, neurological involvement."
  },
  {
    id: "sjogren",
    name: "Sjögren's Syndrome",
    symptoms: "Persistent dry eyes (keratoconjunctivitis sicca), dry mouth (xerostomia), fatigue, joint pain.",
    biomarkers: "Anti-SSA (Ro), Anti-SSB (La).",
    specialist: "Rheumatologist (supported by Ophthalmology/Dentistry)",
    tests: "Schirmer’s test (tear production), Salivary gland biopsy or ultrasound.",
    complications: "Increased risk of lymphoma, dental caries, parotid gland swelling."
  },
  {
    id: "as",
    name: "Ankylosing Spondylitis (AS)",
    symptoms: "Chronic low back pain improving with exercise, morning stiffness, inflammatory eye pain (uveitis).",
    biomarkers: "HLA-B27 (genetic marker, not diagnostic on its own).",
    specialist: "Rheumatologist",
    tests: "Pelvic MRI (detecting sacroiliitis), X-ray of the spine.",
    complications: "Spinal fusion (bamboo spine), restrictive lung disease, aortic valve insufficiency."
  },
  {
    id: "psa",
    name: "Psoriatic Arthritis (PsA)",
    symptoms: "Psoriasis skin lesions, dactylitis (sausage-like fingers), nail pitting, joint/back pain.",
    biomarkers: "Often seronegative (RF/CCP usually negative).",
    specialist: "Rheumatologist, Dermatologist",
    tests: "Imaging (X-ray showing 'pencil-in-cup' changes), skin biopsy if psoriasis is uncertain.",
    complications: "Joint destruction, metabolic syndrome, uveitis."
  },
  {
    id: "ssc",
    name: "Systemic Sclerosis (Scleroderma)",
    symptoms: "Raynaud's phenomenon, skin thickening (fingers/face), acid reflux (GERD), telangiectasia.",
    biomarkers: "Anti-Scl-70, Anti-Centromere, Anti-RNA Polymerase III.",
    specialist: "Rheumatologist",
    tests: "Nailfold capillaroscopy, Pulmonary Function Tests (PFTs), Echocardiogram.",
    complications: "Pulmonary arterial hypertension, interstitial lung disease, scleroderma renal crisis."
  },
  {
    id: "mctd",
    name: "Mixed Connective Tissue Disease (MCTD)",
    symptoms: "Overlap features of SLE, Scleroderma, and Myositis; Raynaud's, swollen 'puffy' hands.",
    biomarkers: "High-titer Anti-U1 RNP (Ribonucleoprotein).",
    specialist: "Rheumatologist",
    tests: "ANA, PFTs, esophageal motility studies.",
    complications: "Pulmonary hypertension, interstitial lung disease."
  }
];
