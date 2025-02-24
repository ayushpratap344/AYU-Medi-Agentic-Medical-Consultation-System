import os
import streamlit as st
import autogen
from autogen import AssistantAgent, GroupChat, GroupChatManager, ConversableAgent
import re

# Configuration
os.environ["AUTOGEN_USE_DOCKER"] = "False"

LLM_CONFIG = {
    "config_list": [{
        "model": "#your_model",
        "base_url": "#your_base_url",
        "api_type": "azure",
        "api_version": "#your_api_version",
        "api_key": "#your_api_key"
    }],
    "temperature": 0.2,
    "max_tokens": 800
}

@st.cache_resource
def create_agents():
    """Create medical team agents"""
    return {
        "Diagnostician": AssistantAgent(
            name="Doctor_Asha",
            system_message="""**Role:** Chief Diagnostician
**Format Requirements:**
### Primary Diagnosis
[Your main diagnosis]

### Differential Diagnoses
1. [Option 1]
2. [Option 2]
3. [Option 3]

### Red Flags
- [Warning sign 1]
- [Warning sign 2]""",
            llm_config=LLM_CONFIG
        ),
        "TreatmentPlanner": AssistantAgent(
            name="Doctor_Ramdev",
            system_message="""**Role:** Treatment Specialist
**Format Requirements:**
### First-line Options
- [Treatment 1]
- [Treatment 2]

### Alternative Therapies
- [Therapy 1]
- [Therapy 2]

### Risk/Benefit Analysis
[Detailed analysis]""",
            llm_config=LLM_CONFIG
        ),
        "EthicsConsultant": ConversableAgent(
            name="EthicsBoard",
            system_message="""Identify ethical considerations. Format:

## Ethical Considerations
- Issue 1
- Issue 2""",
            human_input_mode="NEVER",
            llm_config=LLM_CONFIG
        )
    }

def parse_sections(text):
    """
    Parse the provided text into sections based on headers that start with '###'.
    Returns a dictionary mapping header names (without the ###) to their content.
    Only the first occurrence of each header is kept.
    """
    sections = {}
    header_pattern = re.compile(r'^###\s*(.+?)(?:\s*:)?\s*$', re.MULTILINE)
    matches = list(header_pattern.finditer(text))
    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        if header not in sections:
            sections[header] = text[start:end].strip()
    return sections

def deduplicate_paragraphs(text):
    """
    Remove duplicate paragraphs (blocks separated by two or more newlines)
    while preserving order.
    """
    paragraphs = re.split(r'\n\s*\n', text.strip())
    seen = set()
    unique_paragraphs = []
    for para in paragraphs:
        norm = re.sub(r'\W+', '', para).lower()
        if norm not in seen:
            unique_paragraphs.append(para.strip())
            seen.add(norm)
    return "\n\n".join(unique_paragraphs)

def select_first_occurrence(chat_history, speaker, keyword):
    """
    Return the first message content from chat_history where the message's 'name'
    includes speaker and the content includes the given keyword (case-insensitive).
    """
    for msg in chat_history:
        if speaker.lower() in msg.get("name", "").lower():
            content = msg.get("content", "").strip()
            if re.search(keyword, content, re.IGNORECASE):
                return content
    return ""

def process_chat_history_first(chat_history):
    """
    Process chat history by selecting only the first occurrence of a message containing a
    diagnostic or treatment header.
    Returns a tuple: (diagnostic_text, treatment_text, ethics_text)
    """
    diag_text = select_first_occurrence(chat_history, "Doctor_Asha", "Primary Diagnosis")
    treat_text = select_first_occurrence(chat_history, "Doctor_Ramdev", "First-line Options")
    ethics_text = select_first_occurrence(chat_history, "EthicsBoard", "Ethical Considerations")
    return diag_text, treat_text, ethics_text

def run_consultation(patient_case):
    """Run the medical consultation process."""
    agents = create_agents()
    group_chat = GroupChat(
        agents=list(agents.values()),
        messages=[],
        max_round=2,
        speaker_selection_method="auto"
    )
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=LLM_CONFIG,
        system_message="Coordinate medical team collaboration"
    )
    try:
        result = agents["Diagnostician"].initiate_chat(
            manager,
            message=f"PATIENT CASE:\n{patient_case}",
            max_turns=4
        )
        return result.chat_history
    except Exception as e:
        return [{"error": str(e)}]

# ---------------------------
# Streamlit UI Components
# ---------------------------
st.set_page_config(page_title="MediCollab AI", page_icon="⚕️", layout="wide")
st.title("⚕️ Ayu-Medi Agentic Medical Consultation")
st.markdown("Multi-disciplinary medical analysis platform")

# Move the input form to the left-hand sidebar.
with st.sidebar.form("patient_input"):
    st.header("Patient Input")
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    symptoms = st.text_area("Presenting Symptoms", "Asthma")
    history = st.text_input("Medical History", "Hypertension, smoker")
    submitted = st.form_submit_button("Start Analysis")

if submitted:
    case = f"""**Patient Profile**
- Age: {age}
- Gender: {gender}
- Symptoms: {symptoms}
- Medical History: {history}"""
    
    with st.spinner("Analyzing case with medical AI team..."):
        chat_history = run_consultation(case)
        if "error" in chat_history[0]:
            st.error(f"System Error: {chat_history[0]['error']}")
        else:
            # Select only the first occurrence per group.
            diag_text, treat_text, ethics_text = process_chat_history_first(chat_history)
            # Parse sections from the selected texts.
            diag_sections = parse_sections(diag_text)
            treat_sections = parse_sections(treat_text)
            
            diagnostic_report = {
                "Primary Diagnosis": deduplicate_paragraphs(diag_sections.get("Primary Diagnosis", "No relevant information found.")),
                "Differential Diagnoses": deduplicate_paragraphs(diag_sections.get("Differential Diagnoses", "No relevant information found.")),
                "Red Flags": deduplicate_paragraphs(diag_sections.get("Red Flags", "No relevant information found."))
            }
            treatment_report = {
                "First-line Options": deduplicate_paragraphs(treat_sections.get("First-line Options", "No relevant information found.")),
                "Alternative Therapies": deduplicate_paragraphs(treat_sections.get("Alternative Therapies", "No relevant information found.")),
                "Risk Analysis": deduplicate_paragraphs(treat_sections.get("Risk/Benefit Analysis", "No relevant information found."))
            }
            ethics_report = ethics_text  # Use as-is
            
            md_report = f"""# Medical Analysis Report

## Diagnostic Summary
**Primary Diagnosis:**  
{diagnostic_report.get("Primary Diagnosis")}

**Differential Diagnoses:**  
{diagnostic_report.get("Differential Diagnoses")}

**Red Flags:**  
{diagnostic_report.get("Red Flags")}

## Treatment Plan
**First-line Options:**  
{treatment_report.get("First-line Options")}

**Alternative Therapies:**  
{treatment_report.get("Alternative Therapies")}

**Risk Analysis:**  
{treatment_report.get("Risk Analysis")}

## Ethical Considerations
{ethics_report if ethics_report else "No relevant information found for this section."}
"""
            # Create tabs for final output.
            tab_diag, tab_treat, tab_ethics, tab_raw = st.tabs(
                ["Diagnostic Summary", "Treatment Plan", "Ethical Considerations", "Raw Chat History"]
            )
            with tab_diag:
                st.markdown(f"""# Diagnostic Summary

**Primary Diagnosis:**  
{diagnostic_report.get("Primary Diagnosis")}

**Differential Diagnoses:**  
{diagnostic_report.get("Differential Diagnoses")}

**Red Flags:**  
{diagnostic_report.get("Red Flags")}""")
            with tab_treat:
                st.markdown(f"""# Treatment Plan

**First-line Options:**  
{treatment_report.get("First-line Options")}

**Alternative Therapies:**  
{treatment_report.get("Alternative Therapies")}

**Risk Analysis:**  
{treatment_report.get("Risk Analysis")}""")
            with tab_ethics:
                st.markdown(f"""# Ethical Considerations

{ethics_report if ethics_report else "No relevant information found for this section."}""")
            with tab_raw:
                st.write(chat_history)
            
            st.download_button(
                label="Download Full Report",
                data=md_report,
                file_name="medical_analysis_report.md",
                mime="text/markdown"
            )

st.markdown("---")
st.caption("""
**Disclaimer:** This AI system provides preliminary analysis and should be used by qualified healthcare professionals as a decision support tool only. 
All conclusions require clinical validation. Patient data is processed anonymously in compliance with HIPAA standards.
""")
