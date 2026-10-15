# Explain A Doc

Design a premium, modern clinical intelligence dashboard called MEDIQ.

MEDIQ is an Explainable AI Clinical Decision Support System for
educational research into autoimmune and rheumatic diseases.

DESIGN DIRECTION
Create a polished healthcare SaaS interface inspired by premium
clinical analytics platforms. Use deep navy, teal, white, and
light grey. Use clean typography, generous spacing, subtle borders,
rounded cards, restrained shadows, and professional medical icons.

LAYOUT

1. SIDEBAR
- MEDIQ branding with a medical cross icon
- Dashboard
- Patient Analysis
- Disease Insights
- Explainable AI
- About MEDIQ
- Research prototype status

2. DASHBOARD
- Total supported disease classes
- Number of clinical input features
- XGBoost model information
- SHAP explainability information
- A prominent Start Analysis button

3. PATIENT ANALYSIS
Create a clean form for these 14 features:
Age, Gender, ESR, CRP, RF, Anti-CCP, HLA-B27, ANA,
Anti-Ro, Anti-La, Anti-dsDNA, Anti-Sm, C3 and C4.

Group demographic information, inflammatory markers,
autoantibodies and complement markers logically.

Include a prominent Run Analysis button.

4. PREDICTION RESULTS
- Predicted disease
- Model-reported top-class score
- Horizontal probability distribution for all seven classes
- A clear distinction between the predicted class and other classes

5. EXPLAINABLE AI
- SHAP feature-contribution chart
- Ranked feature-contribution table
- Clear positive and negative contribution indicators
- Explain that SHAP describes model behaviour, not medical causation

6. DISEASE INSIGHTS
- Description
- Common symptoms
- Risk factors
- Important biomarkers
- Diagnostic indicators
- Recommended specialist
- Recommended tests
- Disease category
- Severity levels
- Possible complications

Use readable tabs or clearly separated sections.

7. REPORTS
- Prominent, accessible Download Analysis Report button
- A report preview showing the sections included

IMPORTANT CONSTRAINTS
- This is a frontend design prototype.
- Do not invent patient data or clinical results.
- Do not claim the system is clinically validated.
- Clearly state that model outputs are not medical diagnoses.
- Use realistic placeholder data only for the visual demonstration.
- Make every page responsive and accessible.
- Keep the design professional rather than futuristic or overly decorative.

Generate a complete, visually consistent multi-page frontend.
Prioritise usability, readable charts, clear clinical forms,
and a professional presentation suitable for a university project.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/4047a044-9c97-49af-863d-c6b88cf0d3cf).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
