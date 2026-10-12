from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = {
    "timestamp",
    "depth_m",
    "hdop",
    "num_sats",
}


def run_qc(data, depth_tol, max_gap_s, hdop_max, min_sats):
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    median_depth = data["depth_m"].rolling(window=5, center=True, min_periods=1).median()
    data["depth_flag"] = (data["depth_m"] - median_depth).abs() > depth_tol
    gap_seconds = data["timestamp"].diff().dt.total_seconds()
    data["gap_flag"] = gap_seconds > max_gap_s
    data["gps_flag"] = (data["hdop"] > hdop_max) | (data["num_sats"] < min_sats)
    data["null_flag"] = data["depth_m"].isin([-999, 0]) | data["depth_m"].isna()
    data["any_flag"] = data[["depth_flag", "gap_flag", "gps_flag", "null_flag"]].any(axis=1)
    return data


def make_qc_chart(data):
    figure, axis = plt.subplots(figsize=(12, 5))
    clean_depth = data["depth_m"].where(~data["null_flag"])
    axis.plot(data.index, clean_depth, color="steelblue", linewidth=0.8, label="Depth")
    depth_flags = data["depth_flag"] & ~data["null_flag"]
    axis.scatter(data.index[depth_flags], data.loc[depth_flags, "depth_m"],
                 color="red", s=25, label="Depth spike")
    axis.scatter(data.index[data["gps_flag"]], clean_depth[data["gps_flag"]],
                 color="orange", s=15, label="Poor GPS")
    axis.scatter(data.index[data["gap_flag"]], clean_depth[data["gap_flag"]],
                 color="black", marker="x", s=45, label="Time gap")
    axis.invert_yaxis()
    axis.set_xlabel("Row")
    axis.set_ylabel("Depth (m)")
    axis.set_title("Survey line quality check")
    axis.legend()
    figure.tight_layout()
    return figure


st.set_page_config(page_title="Survey QC Checker", layout="wide")
st.title("Survey QC Checker")

with st.sidebar:
    st.header("QC thresholds")
    depth_tol = st.number_input("Depth deviation (m)", min_value=0.0, value=1.5, step=0.1)
    max_gap_s = st.number_input("Maximum time gap (s)", min_value=0.0, value=1.0, step=0.1)
    hdop_max = st.number_input("Maximum HDOP", min_value=0.0, value=2.0, step=0.1)
    min_sats = st.number_input("Minimum satellites", min_value=0, value=5, step=1)

uploaded_file = st.file_uploader("Upload a survey CSV", type="csv")
default_file = Path(__file__).with_name("survey_line_dummy.csv")

if uploaded_file is not None:
    source = uploaded_file
elif default_file.exists():
    source = default_file
    st.caption(f"Using {default_file.name}. Upload another CSV to check a different survey.")
else:
    source = None
    st.info("Upload a survey CSV to begin.")

if source is not None:
    try:
        survey = pd.read_csv(source)
        missing_columns = REQUIRED_COLUMNS - set(survey.columns)
        if missing_columns:
            st.error("CSV is missing required columns: " + ", ".join(sorted(missing_columns)))
        else:
            for column in ("depth_m", "hdop", "num_sats"):
                survey[column] = pd.to_numeric(survey[column], errors="coerce")
            result = run_qc(survey, depth_tol, max_gap_s, hdop_max, min_sats)

            metric_columns = st.columns(2)
            metric_columns[0].metric("Flagged points", int(result["any_flag"].sum()))
            metric_columns[1].metric("Total points", len(result))
            st.pyplot(make_qc_chart(result))
            st.subheader("Flagged records")
            st.dataframe(result.loc[result["any_flag"]], use_container_width=True)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read the CSV: {error}")