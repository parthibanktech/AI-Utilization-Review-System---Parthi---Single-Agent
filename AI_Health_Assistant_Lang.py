import os
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import json

# ----------------------------------
# ⚙️ Streamlit Setup
# ----------------------------------
st.set_page_config(page_title="🏥Parthi AI Utilization Review", page_icon="💊", layout="wide")
st.title("🏥**AI Utilization Review System - Parthi**")
st.markdown(
    "Easily compare patient data, review medical guidelines, and let the AI agent recommend care decisions."
)

# Custom CSS to reduce text box size and align it left
st.markdown(
    """
    <style>
    .stTextInput input {
        width: 250px;  /* Adjust width as needed */
        max-width: 100%;  /* Ensure it doesn't stretch beyond the container */
        display: block;  /* Make the input field behave like a block element */
        margin-left: 0;  /* Align it to the left */
        margin-right: auto;  /* Remove any auto margin on the right */
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------------------
# 🔑 OpenAI Key
# ----------------------------------
api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
if not api_key:
    st.info("Please enter your OpenAI API key to continue.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# ----------------------------------
# 🧠 Model Setup
# ----------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ----------------------------------
# 🩺 Static Data
# ----------------------------------
medical_guidelines = [
    {"procedure": "MRI Brain", "diagnosis": "Migraine", "required_symptoms": ["headache", "nausea"],
     "notes": "MRI not recommended unless neurological deficits or red flags present."},
    {"procedure": "CT Chest", "diagnosis": "Suspected Pulmonary Embolism", "required_symptoms": ["chest pain", "shortness of breath", "tachycardia"],
     "notes": "CTPA appropriate for high probability PE cases with positive D-dimer."},
    {"procedure": "MRI Lumbar Spine", "diagnosis": "Chronic Low Back Pain", "required_symptoms": ["back pain > 6 weeks", "neurological deficit"],
     "notes": "MRI only if pain persists despite conservative therapy and neuro signs are present."},
    {"procedure": "CT Chest", "diagnosis": "Community-Acquired Pneumonia", "required_symptoms": ["fever", "cough"],
     "notes": "CT Chest reserved for inconclusive X-rays or immunocompromised patients."},
    {"procedure": "CT Abdomen", "diagnosis": "Suspected Appendicitis", "required_symptoms": ["abdominal pain", "nausea", "RLQ tenderness"],
     "notes": "CT imaging justified if appendicitis is unclear."}
]

care_recommendations = [
    {"diagnosis": "Migraine", "next_step": "Start migraine treatment; imaging not necessary unless red flags appear."},
    {"diagnosis": "Suspected Pulmonary Embolism", "next_step": "Begin anticoagulation and confirm with CTPA."},
    {"diagnosis": "Chronic Low Back Pain", "next_step": "Refer to physiotherapy; MRI only if neuro symptoms persist."},
    {"diagnosis": "Community-Acquired Pneumonia", "next_step": "Start empirical antibiotics; reserve CT for poor responders."},
    {"diagnosis": "Suspected Appendicitis", "next_step": "Do CT to confirm and refer for surgery if positive."}
]

patient_records = [
    {"patient_id": "P101", "age": 38, "sex": "Male", "symptoms": ["abdominal pain", "nausea"],
     "diagnosis": "Possible early appendicitis", "procedure": "CT Abdomen",
     "notes": "Mild abdominal pain and nausea but no localized tenderness or rebound noted."},
    {"patient_id": "P102", "age": 65, "sex": "Female", "symptoms": ["chest pain", "shortness of breath", "tachycardia"],
     "diagnosis": "Clinical suspicion of PE", "procedure": "CT Chest",
     "notes": "Wells score high probability; D-dimer positive."},
    {"patient_id": "P103", "age": 30, "sex": "Female", "symptoms": ["recurrent headache"],
     "diagnosis": "Classic migraine presentation", "procedure": "MRI Brain",
     "notes": "No neuro signs or red flags. Typical migraine pattern."}
]

# ----------------------------------
# 🧰 Tools
# ----------------------------------
@tool
def fetch_patient_record(patient_id: str) -> dict:
    """Retrieve structured summary for a given patient."""
    for record in patient_records:
        if record["patient_id"] == patient_id:
            summary = (
                f"Patient ID: {record['patient_id']}\n"
                f"Age: {record['age']}, Sex: {record['sex']}\n"
                f"Reported Symptoms: {', '.join(record['symptoms'])}\n"
                f"Preliminary Diagnosis: {record['diagnosis']}\n"
                f"Requested Procedure: {record['procedure']}\n"
                f"Clinical Notes: {record['notes']}"
            )
            return {"patient_summary": summary}
    return {"error": "Patient record not found."}

@tool
def match_guideline(procedure: str, diagnosis: str) -> dict:
    """Find the closest matching guideline for a given procedure & diagnosis.""" 
    context = "\n".join([f"{g['procedure']} - {g['diagnosis']}: {g['required_symptoms']}" for g in medical_guidelines])
    prompt = f"Procedure: {procedure}\nDiagnosis: {diagnosis}\nAvailable Guidelines:\n{context}\nPick best match."
    result = llm.invoke(prompt).content
    return {"matched_guideline": result}

@tool
def recommend_care_plan(diagnosis: str) -> dict:
    """Recommend next care steps for a given diagnosis."""
    context = "\n".join([f"{c['diagnosis']} → {c['next_step']}" for c in care_recommendations])
    prompt = f"Diagnosis: {diagnosis}\nRecommendations:\n{context}\nPick most appropriate and explain why."
    result = llm.invoke(prompt).content
    return {"recommendation": result}

# ----------------------------------
# 🧠 Agent Definition
# ----------------------------------
single_agent_prompt = """
You are a senior medical review assistant.
Use the available tools to:
- fetch patient record
- match appropriate medical guideline
- recommend care plan

Return your answer in this format:
- **Final Decision:** [APPROVED / NEEDS REVIEW]
- **Reasoning:** [Explain your reasoning]
- **Recommendation:** [Next care plan or alternative]
"""

single_agent = create_react_agent(
    model=llm,
    tools=[fetch_patient_record, match_guideline, recommend_care_plan],
    prompt=SystemMessage(content=single_agent_prompt)
)

# ----------------------------------
# 🧾 Streamlit UI Layout
# ----------------------------------
st.sidebar.header("⚙️ Configuration")
agent_type = st.sidebar.radio("Select Agent Type", ["Single-Agent System"], index=0)
patient_id = st.sidebar.selectbox("🧍 Select Patient ID", [p["patient_id"] for p in patient_records])

# Create layout columns
col1, col2 = st.columns([0.45, 0.55])

# ---- Left side: Patient Info ----
with col1:
    st.subheader("🩺 Patient Record Details")

    # Find the selected patient
    selected = next((p for p in patient_records if p["patient_id"] == patient_id), None)

    if selected:
        # 👁️ Human-friendly view
        st.markdown(f"""
        **Patient ID:** {selected['patient_id']}  
        **Age / Sex:** {selected['age']} / {selected['sex']}  
        **Symptoms:** {", ".join(selected['symptoms'])}  
        **Diagnosis:** {selected['diagnosis']}  
        **Procedure:** {selected['procedure']}  
        **Notes:** {selected['notes']}
        """)

        # 💾 Download as JSON
        st.download_button(
            label="⬇️ Download My Record (JSON)",
            data=json.dumps(selected, indent=2),
            file_name=f"{selected['patient_id']}_record.json",
            mime="application/json",
            use_container_width=True
        )

    else:
        st.warning("No patient record found.")

    # Show all patient data table below for context
    st.markdown("### 📋 All Patient Records")
    st.dataframe(pd.DataFrame(patient_records))

# ---- Right side: AI Review ----
with col2:
    st.subheader("🤖 AI Utilization Review")
    if st.button("Run Review", use_container_width=True):
        with st.spinner("🧠 Running AI Review..."):
            prompt = f"Review patient {patient_id} for medical procedure justification."
            events = single_agent.stream(
                {"messages": [("user", prompt)]},
                {"recursion_limit": 25},
                stream_mode="values"
            )
            final_output = None
            for event in events:
                if "messages" in event:
                    final_output = event["messages"][-1].content

            if final_output:
                st.success("✅ Review Complete!")
                st.markdown("### 📊 **AI Decision Summary**")
                st.markdown(final_output)
            else:
                st.error("⚠️ No response generated. Please try again.")

# ----------------------------------
# 📜 Footer
# ----------------------------------
st.markdown("---")
st.caption("Developed by Parthiban • Powered by 🧠 LangGraph + Streamlit + OpenAI")
