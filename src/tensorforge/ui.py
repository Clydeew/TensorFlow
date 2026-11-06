"""Local UI. Deliberately contains no automatic notebook execution."""

from pathlib import Path

import pandas as pd
import streamlit as st

from tensorforge.cli import doctor
from tensorforge.config import load_config, validate
from tensorforge.data import profile_file
from tensorforge.errors import WorkbenchError
from tensorforge.generator import generate
from tensorforge.registry import catalog

st.set_page_config(page_title="TensorFlow • Baseline workbench", page_icon="◈", layout="wide")
st.markdown(
    """<style>
[data-testid="stAppViewContainer"] {background: #10151d; color: #e5eaf2;}
.block-container {max-width: 1240px; padding-top: 3rem;}
h1 {letter-spacing: -.04em; font-size: 3.4rem !important;}
[data-testid="stMetric"] {border: 1px solid #303b4b; border-radius: 12px; padding: 1rem;}
</style>""",
    unsafe_allow_html=True,
)
st.caption("LOCAL / OFFLINE / HUMAN REVIEWED")
st.title("TensorFlow")
st.write("Turn competition data into an editable baseline notebook.")
st.caption(
    "Independent project. No affiliation with Google's TensorFlow. Generation never trains or runs models."
)
items = catalog()
a, b, c = st.columns(3)
a.metric("Verified core adapters", sum(t["status"] == "verified" for t in items))
b.metric("Experimental templates", sum(t["status"] == "experimental" for t in items))
c.metric("Scenario catalog", len(items))
tabs = st.tabs(["Build a notebook", "Inspect data", "Task coverage", "Environment"])
with tabs[0]:
    st.subheader("Review → validate → generate")
    st.write("Start with an example YAML in the repository. Data paths resolve from its project_root.")
    config_path = st.text_input("Configuration file", "examples/classification.yaml")
    output = st.text_input("New bundle directory", "runs/my-baseline")
    experimental = st.checkbox("Allow experimental local-model templates", value=False)
    left, right = st.columns(2)
    if left.button("Validate configuration", use_container_width=True):
        try:
            cfg, root = load_config(config_path)
            warnings = validate(cfg, root, require_approval=False, experimental=experimental)
            st.success(
                "Schema and local prerequisites validated. Review the warnings before approving the YAML."
            )
            st.json({"task": cfg["task"], "project_root": str(root), "warnings": warnings})
        except (WorkbenchError, OSError, ValueError) as e:
            st.error(str(e))
    if right.button("Generate notebook", type="primary", use_container_width=True):
        try:
            cfg, root = load_config(config_path)
            result = generate(cfg, root, output, experimental=experimental)
            st.success("Notebook generated. No training or inference was performed.")
            st.json(result)
            st.download_button("Download notebook", Path(result["notebook"]).read_bytes(), "baseline.ipynb")
            st.code(f"tensorforge verify --bundle {result['bundle']}", language="bash")
        except (WorkbenchError, OSError, ValueError) as e:
            st.error(str(e))
    st.info(
        "Execution is explicit: open the notebook from the bundle folder or run the verify command in your terminal."
    )
with tabs[1]:
    path = st.text_input("Local CSV, JSONL or Parquet manifest", "examples/data/train.csv")
    limit = st.number_input("Maximum inspected rows", 1, 100000, 1000)
    if st.button("Inspect manifest"):
        try:
            report = profile_file(path, int(limit))
            st.json({k: v for k, v in report.items() if k != "columns"})
            st.dataframe(pd.DataFrame(report["columns"]).T, use_container_width=True)
        except (WorkbenchError, OSError, ValueError) as e:
            st.error(str(e))
with tabs[2]:
    status = st.multiselect(
        "Implementation status",
        ["verified", "experimental", "planned"],
        default=["verified", "experimental", "planned"],
    )
    st.dataframe(
        pd.DataFrame([t for t in items if t["status"] in status])[
            ["id", "title", "status", "backend", "competition_eligibility"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Planned tasks fail explicitly. Experimental templates require compatible local assets and are not declared model-verified."
    )
with tabs[3]:
    st.json(doctor())
    st.markdown(
        "Setup and offline transfer instructions: `README.md` and `docs/PORTABILITY.md` in the repository."
    )
