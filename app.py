import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="MEDIQ | Clinical Intelligence", page_icon="✚", layout="wide")

MODEL_PATH = "mediq_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
KB_PATH = "MEDIQ_Disease_Knowledge_Base.xlsx"

FEATURES = [
    "Age", "Gender", "ESR", "CRP", "RF", "Anti-CCP", "HLA-B27",
    "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm", "C3", "C4"
]
KB_MAPPING = {
    "Rheumatoid Arthritis": "Rheumatoid Arthritis (RA)",
    "Ankylosing Spondylitis": "Ankylosing Spondylitis (AS)",
    "Sjogren Syndrome": "Sjogren's Syndrome (SS)",
    "Psoriatic Arthritis": "Psoriatic Arthritis (PsA)",
    "Reactive Arthritis": "Reactive Arthritis (ReA)",
    "Systemic Lupus Erythematosus": "Systemic Lupus Erythematosus (SLE)",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:#F3F6FB;color:#172B4D}
[data-testid="stSidebar"]{background:#12304A}
[data-testid="stSidebar"] *{color:#EAF4FB}
.brand{display:flex;align-items:center;gap:12px;padding:8px 0 24px}
.brand-icon{width:48px;height:48px;border-radius:14px;display:flex;align-items:center;justify-content:center;color:white;font-size:27px;font-weight:800;background:#16A6A1}
.brand-name{font-size:25px;font-weight:800}.brand-sub{font-size:12px;color:#A8C2D3}
.side-note{border:1px solid #34546A;background:#173B54;border-radius:16px;padding:15px;margin-top:25px;line-height:1.7;font-size:13px}
.title{font-size:clamp(28px,3vw,42px);font-weight:800;letter-spacing:-1.3px;color:#172B4D;margin:5px 0}
.subtitle{color:#718096;font-size:15px;line-height:1.6;margin-bottom:22px}
.banner{background:#E4F1FC;border:1px solid #D1E4F4;color:#244D69;border-radius:18px;padding:17px 20px;margin:14px 0 22px;line-height:1.6}
.card{background:white;border:1px solid #E0E7EF;border-radius:20px;padding:22px;box-shadow:0 5px 18px #12304a0b;margin-bottom:18px}
.metric-label{color:#718096;font-size:14px}.metric-value{font-size:30px;font-weight:800;color:#12304A;margin:8px 0 2px}.metric-caption{color:#7B8794;font-size:12px}
.section-title{font-size:21px;font-weight:750;color:#172B4D;margin-bottom:5px}.section-sub{font-size:13px;color:#718096;line-height:1.6;margin-bottom:15px}
.result{background:linear-gradient(120deg,#12304A,#075766);color:white;border-radius:22px;padding:25px;margin:8px 0 18px}
.eyebrow{color:#73E0D3;font-size:11px;font-weight:800;letter-spacing:1.4px}.disease{font-size:clamp(23px,2.5vw,32px);font-weight:800;margin:10px 0;line-height:1.2}
.confidence{font-size:31px;color:#73E0D3;font-weight:800}.result-caption{color:#D0E2EB;font-size:13px;line-height:1.6}
.pill{display:inline-block;border-radius:999px;padding:5px 10px;background:#DDF6F2;color:#147F7B;font-size:11px;font-weight:700}
.warning{background:#FFF4DF;border:1px solid #F3D7A4;color:#78531A;border-radius:15px;padding:15px 17px;font-size:13px;line-height:1.6;margin:12px 0}
.footer{color:#7B8794;font-size:12px;line-height:1.7;padding:8px 0 20px}
div.stButton>button{border-radius:12px;min-height:44px;font-weight:700;border:1px solid #D6E0E9}
div.stButton>button[kind="primary"]{background:#12304A;color:white;border:1px solid #12304A}
div.stButton>button[kind="primary"]:hover{background:#0B5964;color:white}
div[data-testid="stForm"]{background:white;border:1px solid #E0E7EF;border-radius:20px;padding:20px 22px}
@media(max-width:700px){.card{padding:17px;border-radius:17px}.result{padding:20px}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH), pd.read_excel(KB_PATH)

try:
    model, le, kb = load_assets()
except Exception as e:
    st.error("Could not load the model, label encoder, or Excel knowledge base. Keep all three files beside app.py.")
    st.code(str(e))
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">✚</div><div><div class="brand-name">MEDIQ</div><div class="brand-sub">Clinical Intelligence</div></div></div><p style="color:#8FAFC3;font-size:11px;letter-spacing:1.5px">WORKSPACE</p>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Dashboard", "Patient Analysis", "Disease Insights", "Explainable AI", "About MEDIQ"], label_visibility="collapsed")
    st.markdown('<div class="side-note"><strong>● Research Prototype</strong><br>Model interface · MEDIQ v1.0<br><span style="color:#A8C2D3">Educational use, not a clinical diagnosis.</span></div>', unsafe_allow_html=True)

st.markdown('<div class="title">Clinical Intelligence Dashboard</div><div class="subtitle">AI-assisted analysis of autoimmune and rheumatic disease patterns.</div><div class="banner">ⓘ &nbsp; MEDIQ analyzes clinical biomarkers and provides explainable model predictions to support research and clinical review.</div>', unsafe_allow_html=True)

def find_kb_row(disease):
    if "Disease" not in kb.columns:
        return None
    target = KB_MAPPING.get(disease, disease)
    rows = kb[kb["Disease"].astype(str).str.strip().str.casefold() == target.strip().casefold()]
    if rows.empty:
        rows = kb[kb["Disease"].astype(str).str.strip().str.casefold() == disease.strip().casefold()]
    return None if rows.empty else rows.iloc[0]

def safe_text(v):
    return "Information is not available in the current knowledge base." if pd.isna(v) or str(v).strip().lower() in ("", "nan") else str(v)

def predict(patient):
    raw = model.predict(patient)[0]
    disease = str(le.inverse_transform([raw])[0])
    probabilities = model.predict_proba(patient)[0]
    class_values = list(model.classes_)
    rows = []
    for i, c in enumerate(class_values):
        name = str(le.inverse_transform([c])[0])
        rows.append({"Disease": name, "Model output (%)": float(probabilities[i]) * 100})
    probs = pd.DataFrame(rows).sort_values("Model output (%)", ascending=False)
    class_index = class_values.index(raw)
    return disease, float(np.max(probabilities)) * 100, probs, class_index

def explain(patient, class_index):
    vals = shap.TreeExplainer(model).shap_values(patient)
    vals = np.asarray(vals)
    if vals.ndim == 3:
        contributions = vals[0, :, class_index]
    elif vals.ndim == 2:
        contributions = vals[0, :]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {vals.shape}")
    df = pd.DataFrame({"Feature": FEATURES, "SHAP contribution": np.asarray(contributions, dtype=float)})
    df["Absolute contribution"] = df["SHAP contribution"].abs()
    return df.sort_values("Absolute contribution", ascending=False)

if page == "Dashboard":
    cols = st.columns(4)
    cards = [("Disease Categories","7","Supported prediction classes"),("Clinical Features","14","Patient input variables"),("Model","XGBoost","Machine learning classifier"),("Explainability","SHAP","Feature contribution analysis")]
    for c, (label, value, caption) in zip(cols, cards):
        with c:
            st.markdown(f'<div class="card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-caption">{caption}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="result"><div class="eyebrow">START AN ANALYSIS</div><div class="disease">Evaluate a patient’s biomarker profile</div><div class="result-caption">Enter 14 clinical variables to explore the model’s ranked outputs with transparent feature contributions.</div></div>', unsafe_allow_html=True)
    st.info("Choose **Patient Analysis** in the sidebar to enter clinical variables and run the model.")
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="card"><div class="section-title">Model inputs</div><div class="section-sub">Age · Gender · ESR · CRP · RF · Anti-CCP · HLA-B27 · ANA · Anti-Ro · Anti-La · Anti-dsDNA · Anti-Sm · C3 · C4</div></div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="section-title">Research mode</div><div class="section-sub">Educational prototype; not clinically validated.</div><span class="pill">Model + SHAP</span></div>',unsafe_allow_html=True)

elif page == "Patient Analysis":
    st.markdown('<div class="section-title">Patient Analysis</div><div class="section-sub">Enter de-identified clinical variables to evaluate the model-supported disease classes.</div><div class="banner">ⓘ &nbsp; Use de-identified data only. Results are research outputs to support — not replace — clinical judgement.</div>',unsafe_allow_html=True)
    with st.form("mediq_form"):
        st.markdown('<div class="section-title">Clinical input</div><div class="section-sub">Demographics, laboratory markers, and autoantibodies</div>',unsafe_allow_html=True)
        st.caption("Gender retains the model's original 0/1 encoding. Verify the training-data mapping before interpreting the values.")
        c1,c2=st.columns(2)
        with c1:
            age=st.number_input("Age (years)",min_value=1,max_value=120,value=30)
            gender=st.selectbox("Gender encoding",[0,1],help="Uses the original model encoding; confirm which value represents each gender.")
            esr=st.number_input("ESR",min_value=0.0,value=20.0,step=1.0)
            crp=st.number_input("CRP",min_value=0.0,value=5.0,step=0.5)
            rf=st.number_input("Rheumatoid Factor (RF)",min_value=0.0,value=10.0,step=1.0)
            c3=st.number_input("Complement C3",min_value=0.0,value=100.0,step=1.0)
            c4=st.number_input("Complement C4",min_value=0.0,value=20.0,step=1.0)
        with c2:
            binary_labels={}
            for feature in ["Anti-CCP","HLA-B27","ANA","Anti-Ro","Anti-La","Anti-dsDNA","Anti-Sm"]:
                binary_labels[feature]=st.selectbox(feature,["Negative (0)","Positive (1)"],index=0)
        st.caption("Binary biomarker selections map Negative to 0 and Positive to 1. Confirm this matches the training pipeline.")
        submitted=st.form_submit_button("▶  Run Analysis",type="primary",use_container_width=True)
    if submitted:
        binary={k:(1 if v.startswith("Positive") else 0) for k,v in binary_labels.items()}
        # Exact trained model order — do not reorder these values.
        patient_values=[age,gender,esr,crp,rf,binary["Anti-CCP"],binary["HLA-B27"],binary["ANA"],binary["Anti-Ro"],binary["Anti-La"],binary["Anti-dsDNA"],binary["Anti-Sm"],c3,c4]
        patient=np.asarray([patient_values],dtype=float)
        try:
            with st.spinner("Analyzing clinical features..."):
                disease,confidence,probs,class_index=predict(patient)
                shap_df=explain(patient,class_index)
            st.session_state["mediq_result"]={"disease":disease,"confidence":confidence,"probs":probs,"shap":shap_df,"values":patient_values,"time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        except Exception as e:
            st.error("Analysis failed. Check model compatibility and SHAP output.")
            st.code(str(e))
    result=st.session_state.get("mediq_result")
    if result:
        disease=result["disease"]; confidence=result["confidence"]; probs=result["probs"]; shap_df=result["shap"]
        st.markdown(f'<div class="result"><div class="eyebrow">MOST LIKELY MODEL PATTERN</div><div class="disease">{disease}</div><div class="confidence">{confidence:.1f}%</div><span class="pill">Model-reported confidence</span><div class="result-caption" style="margin-top:12px">Analysis generated {result["time"]}. This is a model output, not a confirmed diagnosis.</div></div>',unsafe_allow_html=True)
        if disease=="Normal":
            st.info("The model returned the Normal class. This does not rule out an underlying condition.")
        left,right=st.columns([1.1,1])
        with left:
            st.markdown('<div class="card"><div class="section-title">Probability distribution</div><div class="section-sub">Model outputs for the seven supported classes.</div>',unsafe_allow_html=True)
            st.bar_chart(probs.set_index("Disease"),horizontal=True,height=300)
            st.markdown('<div class="section-sub">These values are not necessarily calibrated clinical probabilities.</div></div>',unsafe_allow_html=True)
        with right:
            st.markdown('<div class="card"><div class="section-title">Top contributing factors</div><div class="section-sub">SHAP contributions for the predicted class.</div>',unsafe_allow_html=True)
            top=shap_df.head(5).sort_values("SHAP contribution")
            fig,ax=plt.subplots(figsize=(6,3.5)); ax.barh(top["Feature"],top["SHAP contribution"]); ax.axvline(0,linewidth=.8); ax.set_xlabel("SHAP contribution"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout()
            st.pyplot(fig,use_container_width=True); plt.close(fig)
            st.markdown('<div class="section-sub">SHAP describes model behavior, not medical causation.</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="section-title">Disease Insights</div><div class="section-sub">Information from the existing MEDIQ knowledge base.</div>',unsafe_allow_html=True)
        row=find_kb_row(disease)
        if disease=="Normal":
            st.info("The knowledge base may not include a disease profile for the Normal class.")
        elif row is None:
            st.warning(f"No matching knowledge-base entry found for {disease}.")
        else:
            tab_defs=[("Overview","Description"),("Symptoms","Common Symptoms"),("Biomarkers","Important Biomarkers"),("Specialist","Recommended Specialist"),("Tests","Recommended Tests"),("Complications","Possible Complications")]
            tabs=st.tabs([x[0] for x in tab_defs])
            for tab,(_,col) in zip(tabs,tab_defs):
                with tab:
                    st.write(safe_text(row[col]) if col in kb.columns else f"The '{col}' column is not present in the knowledge base.")
        report=["MEDIQ — Clinical Intelligence Analysis Report",f"Timestamp: {result['time']}","",f"Predicted class: {disease}",f"Model-reported confidence: {confidence:.2f}%","","Patient input values:"]
        report += [f"- {k}: {v}" for k,v in zip(FEATURES,result["values"])]
        report += ["","Model output distribution:"]+[f"- {r['Disease']}: {r['Model output (%)']:.2f}%" for _,r in probs.iterrows()]
        report += ["","Top SHAP contributions:"]+[f"- {r['Feature']}: {r['SHAP contribution']:+.4f}" for _,r in shap_df.head(5).iterrows()]
        report += ["","Disclaimer: Educational/research prototype only. Not a medical diagnosis. Consult a qualified healthcare professional."]
        st.download_button("⬇  Download Analysis Report","\n".join(report),file_name="MEDIQ_Analysis_Report.txt",mime="text/plain",use_container_width=True)
        st.markdown('<div class="warning"><strong>Not a diagnosis.</strong> This research model output indicates a predicted pattern. Clinical confirmation requires evaluation by a qualified healthcare professional.</div>',unsafe_allow_html=True)

elif page == "Disease Insights":
    st.markdown('<div class="section-title">Disease Insights</div><div class="section-sub">Seven classes supported by the current trained model.</div>',unsafe_allow_html=True)
    diseases=[("RA","Rheumatoid Arthritis"),("AS","Ankylosing Spondylitis"),("PsA","Psoriatic Arthritis"),("ReA","Reactive Arthritis"),("SS","Sjogren Syndrome"),("SLE","Systemic Lupus Erythematosus"),("N","Normal")]
    for i in range(0,len(diseases),2):
        cols=st.columns(2)
        for col,(abbr,name) in zip(cols,diseases[i:i+2]):
            with col:
                row=find_kb_row(name)
                desc="No disease profile is applicable to the Normal class."
                if row is not None and "Description" in kb.columns: desc=safe_text(row["Description"])
                st.markdown(f'<div class="card"><span class="pill">{abbr}</span><div class="section-title" style="margin-top:12px">{name}</div><div class="section-sub">{desc}</div></div>',unsafe_allow_html=True)

elif page == "Explainable AI":
    st.markdown('<div class="section-title">Explainable AI</div><div class="section-sub">Understand which features contributed most to the model output.</div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="section-title">How to read SHAP</div><p>Features with larger absolute SHAP values have a larger contribution to the model output for the prediction being explained. Direction and magnitude depend on the model and class.</p><p>These values describe model behavior. They do not establish medical causation or clinical importance.</p></div>',unsafe_allow_html=True)
    result=st.session_state.get("mediq_result")
    if result:
        st.write("Last analyzed class:",result["disease"])
        st.dataframe(result["shap"][["Feature","SHAP contribution","Absolute contribution"]],use_container_width=True,hide_index=True)
    else: st.info("Run a patient analysis to see SHAP contributions for a specific prediction.")

elif page == "About MEDIQ":
    st.markdown('<div class="section-title">About MEDIQ</div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="section-title">Explainable AI Clinical Decision Support System</div><p>MEDIQ is an educational research prototype exploring machine-learning classification and explainability for a limited set of autoimmune and rheumatic disease categories.</p><p><strong>Classifier:</strong> XGBoost<br><strong>Input features:</strong> 14 clinical variables<br><strong>Explanation method:</strong> SHAP<br><strong>Knowledge base:</strong> Excel-based disease information</p></div><div class="warning"><strong>Research-use disclaimer:</strong> MEDIQ is not a clinically validated diagnostic system. Predictions must not be used alone to make medical decisions. Model confidence is not necessarily a calibrated probability of disease.</div>',unsafe_allow_html=True)

st.markdown('<div class="footer">MEDIQ — Explainable AI Clinical Decision Support System<br>Educational research prototype. Outputs are not medical diagnoses and must be reviewed by qualified clinicians.</div>',unsafe_allow_html=True)
