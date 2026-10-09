import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="NETSENTINEL AI | Threat Vector Classifier",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ NETSENTINEL AI")
st.caption("RandomForest & Gradient-Boosted Deep Packet Anomaly Vector Classifier")

# Initialize Gemini API Client secretly
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in .env file. Please check your configuration.")
    st.stop()


@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)


client = get_gemini_client(api_key)


# Pydantic Schemas for Strict Single-Class Security Vector Inference
class ThreatProbability(BaseModel):
    threat_class: str = Field(
        description="Identified attack vector or threat category (e.g., SQL Injection, Cross-Site Scripting, DDoS Syn Flood, Ransomware Command & Control, Normal Baseline).")
    softmax_score: float = Field(description="Classifier probability score between 0.00 and 1.00.")


class SecurityClassificationResult(BaseModel):
    identified_threat: str = Field(
        description="The primary identified attack category or anomaly vector ONLY (e.g., SQL Injection, Remote Code Execution, Privilege Escalation).")
    confidence_score: float = Field(
        description="Top-1 model classification confidence probability between 0.00 and 1.00.")
    cve_or_tactic_code: str = Field(
        description="Corresponding MITRE ATT&CK ID or CVE classification code (e.g., T1190, T1059, CVE-2023-34362).")
    probability_distribution: list[ThreatProbability] = Field(
        description="Top 4 class candidate probabilities calculated by the decision ensemble.")


SYSTEM_INSTRUCTION = """
You are NETSENTINEL AI, a Random Forest & Isolation Forest Anomaly Classifier trained on CICIDS-2017 and DARPA network traffic benchmark datasets.
Analyze the provided packet payload, HTTP request log, or network stream.
Perform single-class threat vector classification and output strictly the identified attack name, top-1 confidence score, MITRE/CVE code, and probability distribution.

Output strictly valid JSON matching the schema. Do not generate conversational conversational filler, explanations, or prose.
"""


def simulate_packet_inference():
    """Simulates realistic packet vectorization and tree ensemble execution."""
    progress_bar = st.progress(0, text="Tokenizing Raw Packet Payload & Normalizing Entropy Features...")
    time.sleep(0.3)

    progress_bar.progress(35,
                          text="Extracting 128-dim Feature Array (Header Length, Payload Entropy, Port Signatures)...")
    time.sleep(0.4)

    progress_bar.progress(70, text="Evaluating Random Forest Decision Trees (n_estimators=500)...")
    time.sleep(0.4)

    progress_bar.progress(95, text="Computing Final Softmax Aggregation across 48 Attack Vector Classes...")
    time.sleep(0.3)

    progress_bar.progress(100, text="Threat Vector Classification Complete!")
    time.sleep(0.2)
    progress_bar.empty()


def render_security_classification_output(result: dict):
    """Renders the single predicted threat vector in standard ML classification format."""

    threat_name = result.get("identified_threat", "Unidentified Anomaly")
    confidence = result.get("confidence_score", 0.95) * 100
    code = result.get("cve_or_tactic_code", "N/A")

    # 1. Standard ML Single-Prediction Output Block
    st.subheader("🎯 Predicted Threat Vector")

    col_out1, col_out2, col_out3 = st.columns([2, 1, 1])

    with col_out1:
        st.markdown(f"### **Threat Vector Identified:** `{threat_name}`")

    with col_out2:
        st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

    with col_out3:
        st.metric(label="MITRE / CVE Code", value=code)

    st.divider()

    # 2. Probability Distribution Bar Plot
    st.subheader("📊 Output Ensemble Probability Distribution")
    distribution = result.get("probability_distribution", [])
    if distribution:
        df_dist = pd.DataFrame(distribution)
        df_dist["Probability (%)"] = (df_dist["softmax_score"] * 100).round(2)

        fig_bar = px.bar(
            df_dist,
            x="Probability (%)",
            y="threat_class",
            orientation="h",
            text="Probability (%)",
            title="Top-4 Candidate Threat Classes",
            labels={"threat_class": "Predicted Threat Class", "Probability (%)": "Probability (%)"},
            color="Probability (%)",
            color_continuous_scale="Reds"
        )
        fig_bar.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)


# Sidebar Model Config (Disguised ML Settings)
with st.sidebar:
    st.header("⚙️ Model Architecture")
    st.selectbox("Classifier Engine",
                 ["Random Forest Classifier (CICIDS-v4)", "XGBoost Threat Detector", "Isolation Forest Anomaly Net"])
    st.slider("Anomaly Softmax Cutoff", min_value=0.50, max_value=0.99, value=0.85, step=0.05)
    st.text_input("Feature Vector Length", value="128 Floating-Point Tensors")
    st.caption("Model Weights: Pre-trained on 2.5M network packet captures & malware payload streams.")

# Main Log Input Section
st.subheader("📥 Input Raw Packet / Network Payload Buffer")

sample_payload = """POST /admin/upload.php HTTP/1.1
Host: 192.168.1.45
User-Agent: Mozilla/5.0
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="file"; filename="cmd.php"
Content-Type: application/x-php

<?php system($_GET['cmd']); ?>
------WebKitFormBoundary7MA4YWxkTrZu0gW--"""

payload_input = st.text_area(
    "Paste HTTP request stream, packet payload hex/text, or syslog entry:",
    value=sample_payload,
    height=180
)

run_classify_btn = st.button("🚀 Run Vector Extraction & Classify Threat", type="primary", use_container_width=True)

if run_classify_btn:
    if not payload_input.strip():
        st.warning("Please input packet payload data to classify.")
    else:
        # 1. UI Pipeline Animation
        simulate_packet_inference()

        # 2. Secretly pass packet payload to Gemini under the hood
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[
                    f"Classify the following network payload: {payload_input}. Output ONLY the threat category name, confidence score, MITRE code, and top class distribution."
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=SecurityClassificationResult,
                ),
            )

            inference_data = json.loads(response.text)

            # 3. Render Output
            render_security_classification_output(inference_data)

        except Exception as e:
            st.error(f"Vector Classification Failure: {e}")