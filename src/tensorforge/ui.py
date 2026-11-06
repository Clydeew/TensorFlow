"""Local UI. Deliberately contains no automatic notebook execution."""

from pathlib import Path

import pandas as pd
import streamlit as st

from tensorforge.cli import doctor
from tensorforge.config import load_config, validate
from tensorforge.data import profile_file
from tensorforge.errors import WorkbenchError
from tensorforge.generator import generate
from tensorforge.languages import PRESETS
from tensorforge.registry import catalog
from tensorforge.ui_text import ENGLISH

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
t = ENGLISH
st.caption(t["badge"])
st.title("TensorFlow")
st.write(t["intro"])
st.caption(t["independent"])
items = catalog()
a, b, c = st.columns(3)
a.metric(t["verified"], sum(t["status"] == "verified" for t in items))
b.metric(t["experimental"], sum(t["status"] == "experimental" for t in items))
c.metric(t["catalog"], len(items))
tabs = st.tabs(t["tabs"])
with tabs[0]:
    st.subheader(t["workflow"])
    st.write(t["start"])
    config_path = st.text_input(t["config"], "examples/indonesia.yaml")
    output = st.text_input(t["output"], "runs/my-baseline")
    experimental = st.checkbox(t["allow"], value=False)
    left, right = st.columns(2)
    if left.button(t["validate"], use_container_width=True):
        try:
            cfg, root = load_config(config_path)
            warnings = validate(cfg, root, require_approval=False, experimental=experimental)
            st.success(t["valid"])
            st.json({"task": cfg["task"], "project_root": str(root), "warnings": warnings})
        except (WorkbenchError, OSError, ValueError) as e:
            st.error(str(e))
    if right.button(t["generate"], type="primary", use_container_width=True):
        try:
            cfg, root = load_config(config_path)
            result = generate(cfg, root, output, experimental=experimental)
            st.success(t["generated"])
            st.json(result)
            st.download_button(t["download"], Path(result["notebook"]).read_bytes(), "baseline.ipynb")
            st.code(f"tensorforge verify --bundle {result['bundle']}", language="bash")
        except (WorkbenchError, OSError, ValueError) as e:
            st.error(str(e))
    st.info(t["execution"])
with tabs[1]:
    path = st.text_input(t["manifest"], "examples/data/indonesia_train.csv")
    limit = st.number_input(t["limit"], 1, 100000, 1000)
    if st.button(t["inspect"]):
        try:
            report = profile_file(path, int(limit))
            st.json({k: v for k, v in report.items() if k != "columns"})
            st.dataframe(pd.DataFrame(report["columns"]).T, use_container_width=True)
        except (WorkbenchError, OSError, ValueError) as e:
            st.error(str(e))
with tabs[2]:
    status = st.multiselect(
        t["status"],
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
    st.caption(t["coverage_note"])
with tabs[3]:
    st.subheader(t["languages"])
    st.write(t["language_note"])
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "tag": tag,
                    "language": info["name"],
                    "scripts": ", ".join(info["scripts"]),
                    "ocr_code": info["ocr"] or "—",
                }
                for tag, info in PRESETS.items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.json(doctor())
    st.markdown(t["setup"])
