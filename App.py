import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

st.write("Hello")
st.title("Science App ")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    st.write("✅ File uploaded successfully!")
    if uploaded_file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, low_memory=False)
    # df = pd.read_csv(uploaded_file)
    # Drop empty or unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  # remove columns like 'Unnamed: 79'
    df = df.dropna(axis=1, how="all")  # also drop any all-NaN columns just in case


    st.subheader("Data Overview")
    st.dataframe(df.head())

    if st.button("Show Summary Statistics"):
        st.write("🚩 Summary Statistics:")
        st.write(df.describe())

    # --- Helper Functions for Normalization ---
    def normalize_visit_column(df, visit_col):
        if visit_col in df.columns:
            df[visit_col] = (df[visit_col].astype(str)
                             .str.strip()
                             .str.replace(r"\s+", " ", regex=True)
                             .str.title())
        return df

    def normalize_pass_fail_columns(df, visit_col):
        def normalize_pf(s):
            s = s.astype("string").str.strip().str.lower()
            s = s.replace({
                "passed": "pass", "ok": "pass", "true": "pass", "1": "pass", "y": "pass", "yes": "pass",
                "failed": "fail", "false": "fail", "0": "fail", "n": "fail", "no": "fail",
                "": pd.NA, "na": pd.NA, "n/a": pd.NA, "nan": pd.NA, "-": pd.NA, "--": pd.NA, ".": pd.NA
            })
            return s

        for c in df.select_dtypes(include=["object", "string", "category"]).columns:
            if c != visit_col:
                df[c] = normalize_pf(df[c])
        return df

    # --- Sidebar for User Configuration ---
    st.sidebar.header("⚙️ Analysis Configuration")

    all_columns = df.columns.tolist()
    visit_col = st.sidebar.selectbox("Select Visit Column", all_columns, index=all_columns.index("VISIT") if "VISIT" in all_columns else 0)
    subjid_col = st.sidebar.selectbox("Select Subject ID Column", all_columns, index=all_columns.index("SUBJID") if "SUBJID" in all_columns else 0)

    df = normalize_visit_column(df, visit_col)
    all_visits = sorted(df[visit_col].dropna().unique())
    default_visits = [v for v in ["Screening", "Week52", "Week80"] if v in all_visits]
    selected_visits = st.sidebar.multiselect("Select Visits for Analysis", all_visits, default=default_visits or all_visits)

    # For filtering
    st.subheader("🚩 1. Filter Data")
    columns = df.columns.tolist()
    selected_column = st.selectbox("Select Column to Filter by", columns)

    # Avoid trying to select before column is chosen
    if selected_column:
        unique_values = df[selected_column].dropna().unique().tolist()
        selected_value = st.selectbox("Select Value", unique_values)

        if selected_value:
            filtered_df = df[df[selected_column] == selected_value]
            st.write(f"Showing rows where `{selected_column}` = **{selected_value}**")
            st.dataframe(filtered_df)
            st.write(f"Total rows after filtering: {filtered_df.shape[0]}")

    # -------------------------------
    # QC Pass Rate Analysis Section
    # -------------------------------
   

    st.subheader("🚩 2. QC Pass Rate Analysis")

    if st.button("Run QC Pass Rate Analysis"):
        df_qc = df.copy()  # use uploaded dataframe

        # 1) Normalize data
        df_qc = normalize_pass_fail_columns(df_qc, visit_col)

        # 3) Detect QC columns
        qc_cols = [c for c in df_qc.columns
                if c != visit_col
                and df_qc[c].dtype in ["object","string","category"]
                and set(df_qc[c].dropna().unique()).issubset({"pass","fail"})
                and df_qc[c].notna().any()]

        if not qc_cols:
            st.warning("No QC columns found in file.")
        
        else:
            # Let user select the primary QC gate
            default_target_qc = "MRORRES IMVOLQC" if "MRORRES IMVOLQC" in qc_cols else qc_cols[0]
            target_col = st.selectbox("Select Primary QC Gate Column", qc_cols, index=qc_cols.index(default_target_qc))

            # 4) Perform computations
            desired_visits = selected_visits

            def compute_rates(subframe, cols):
                valid = subframe[cols].isin(["pass","fail"])
                den = valid.groupby(subframe["VISIT"]).sum().reindex(desired_visits)
                num = subframe[cols].eq("pass").groupby(subframe["VISIT"]).sum().reindex(desired_visits)
                pct = (num / den.replace(0, np.nan) * 100).round(1).fillna(0)
                return num, den, pct

            sub = df_qc[df_qc[visit_col].isin(desired_visits)]
            num_target, den_target, pct_target = compute_rates(sub, [target_col])

            other_cols = [c for c in qc_cols if c != target_col]
            sub_pass = sub[sub[target_col] == "pass"]
            num_other, den_other, pct_other = compute_rates(sub_pass, other_cols)

            friendly_names = {
                "MRORRES IMVOLQC": "Volumetric Suitability",
                "MRORRES WLWBVQC": "LEAP Whole Brain",
                "MRORRES VLLVQC": "LEAP Lateral Ventricles",
                "MRORRES HLHQCC": "LEAP Hippocampus",
                "MRORRES MTLLQC": "LEAP Medial Temporal Lobe",
                "MRORRES PTIVQC": "ASF or pTIV"
            }

            ordered_cols = [target_col] + other_cols
            wide = pd.DataFrame({"QC Column": [friendly_names.get(c, c) for c in ordered_cols]})
            for v in desired_visits:
                pct_vals, count_vals = [], []
                for c in ordered_cols:
                    if c == target_col:
                        n, d, p = num_target.loc[v, c], den_target.loc[v, c], pct_target.loc[v, c]
                    else:
                        n, d, p = num_other.get(c, pd.Series()).get(v, np.nan), den_other.get(c, pd.Series()).get(v, 0), pct_other.get(c, pd.Series()).get(v, 0)
                    n_int, d_int = (0 if pd.isna(n) else int(n)), (0 if pd.isna(d) else int(d))
                    pct_vals.append(p)
                    count_vals.append(f"{n_int}/{d_int}")
                wide[f"{v} %"] = pct_vals
                wide[f"{v} (pass/total)"] = count_vals

            st.dataframe(wide)

            # Allow Excel download
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                wide.to_excel(writer, sheet_name="QC_PassRate", index=False)
            st.download_button("Download QC Results", buf.getvalue(), "QC_PassRate.xlsx", "application/vnd.ms-excel")
    # -------------------------------
    # MRORRES Statistics (QC-pass only)
    # -------------------------------
    st.subheader("🚩 3. Endpoint Statistics (QC-pass only)")

    if st.button("Run MRORRES Statistics"):
        df_stats = df.copy()

        # ---- Normalize data ----
        df_stats = normalize_pass_fail_columns(df_stats, visit_col)

        # ---- Create friendly names for measures ----
        import re
        code_to_label = {}
        for c in [c for c in df_stats.columns if str(c).upper().startswith("MRITEST")]:
            code = re.sub(r'^\s*(MRORRES|MRITEST)\s*', '', str(c), flags=re.I).strip().upper()
            ser = df_stats[c].dropna().astype(str).str.strip()
            if not ser.empty:
                mode_vals = ser[ser != ""].mode()
                label = mode_vals.iat[0] if not mode_vals.empty else ser.iloc[0]
                code_to_label[code] = label

        # Identify numeric and QC columns
        all_cols = df_stats.columns
        numeric_cols, num_df = [], {visit_col: df_stats[visit_col]}
        measure_prefix = st.text_input("Enter prefix for numeric measures (e.g., MRORRES)", "MRORRES")
        for c in [col for col in all_cols if str(col).upper().startswith(measure_prefix.upper())]:
            col_num = pd.to_numeric(df_stats[c], errors="coerce")
            if col_num.notna().sum() > 1: # Require at least 2 numeric values
                numeric_cols.append(c)
                num_df[c] = col_num
        num_df = pd.DataFrame(num_df)

        qc_cols = [c for c in mrorres_cols
                if set(df_stats[c].dropna().unique()).issubset({"pass", "fail"})
                and df_stats[c].notna().any()]

        default_target_qc = "MRORRES IMVOLQC" if "MRORRES IMVOLQC" in qc_cols else (qc_cols[0] if qc_cols else None)
        target_col = st.selectbox("Select Primary QC Gate for Statistics", qc_cols, index=qc_cols.index(default_target_qc) if default_target_qc else 0, key="stats_qc_gate")

        if target_col not in qc_cols:
            st.warning(f"Required QC gate column '{target_col}' not found or not pass/fail type.")
        else:
            # Simplified QC mapping: look for MeasureNameQC or MeasureName_QC
            def qc_for_measure(meas_col: str):
                for suffix in ["QCC", "QC", "_QC"]:
                    potential_qc_col = f"{meas_col}{suffix}"
                    if potential_qc_col in qc_cols:
                        return potential_qc_col
                # Fallback for more complex names like MRORRES WLWBVQC for MRORRES WLLWBV
                base_name = str(meas_col).replace(measure_prefix, "").strip()
                for qc_col in qc_cols:
                    if base_name in qc_col:
                        return qc_col
                return None

            num2qc = {c: qc_for_measure(c) for c in numeric_cols}
            numeric_cols = [c for c in numeric_cols if num2qc[c] is not None]
            visit_order = selected_visits

            def stats_for(meas_col: str, visit: str):
                qcol_meas = num2qc[meas_col]
                mask = (
                    (df_stats[visit_col] == visit)
                    & (df_stats[target_col] == "pass")
                    & (df_stats[qcol_meas] == "pass")
                )
                vals = num_df.loc[mask, meas_col]
                if vals.empty:
                    return pd.Series({"Mean": np.nan, "SD": np.nan, "Min": np.nan, "Max": np.nan, "N": 0})
                return pd.Series({
                    "Mean": vals.mean(),
                    "SD": vals.std(ddof=1),
                    "Min": vals.min(),
                    "Max": vals.max(),
                    "N": int(vals.notna().sum())
                })

            rows = []
            for meas in numeric_cols:
                row = {"Measure": code_to_label.get(re.sub(r'^\s*(MRORRES|MRITEST)\s*', '', str(meas), flags=re.I).strip().upper(), meas)}
                for v in visit_order:
                    s = stats_for(meas, v)
                    row[f"{v} Mean"] = s["Mean"]
                    row[f"{v} SD"] = s["SD"]
                    row[f"{v} Min"] = s["Min"]
                    row[f"{v} Max"] = s["Max"]
                    row[f"{v} N"] = s["N"]
                rows.append(row)

            wide = pd.DataFrame(rows)
            for c in wide.columns:
                if c.endswith((" Mean", " SD", " Min", " Max")):
                    wide[c] = wide[c].round(3)

            st.dataframe(wide)

            # Excel download
            buf_stats = io.BytesIO()
            with pd.ExcelWriter(buf_stats, engine="openpyxl") as writer:
                wide.to_excel(writer, sheet_name="MRORRES_Stats_QC_Pass", index=False)

            st.download_button(
                label="Download MRORRES Statistics",
                data=buf_stats.getvalue(),
                file_name="MRORRES_Stats_QC_Pass.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # -------------------------------
    # Outlier Detection (>3 SD), QC-pass only (needs SUBJID)
    # -------------------------------
    st.subheader("🚩 4. Outlier Detection (> 3 SD) — QC-pass only")

    if st.button("Run Outlier Detection"):
        if subjid_col not in df.columns:
            st.error(f"The dataset must contain a '{subjid_col}' column to track subjects.")
            st.stop()

        df_ol = df.copy()

        # -- Normalize data
        df_ol = normalize_pass_fail_columns(df_ol, visit_col)

        # -- Build friendly names from MRITEST
        import re
        def extract_code(name: str) -> str:
            return re.sub(r'^\s*(MRORRES|MRITEST)\s*', '', str(name), flags=re.I).strip().upper()

        code_to_label = {} # Re-using from previous step if run, but safer to rebuild
        for c in [c for c in df_ol.columns if "TEST" in str(c).upper()]:
            code = extract_code(c)
            ser = df_ol[c].dropna().astype(str).str.strip()
            if not ser.empty:
                m = ser[ser != ""].mode()
                code_to_label[code] = (m.iat[0] if not m.empty else ser.iloc[0])

        # -- MRORRES numeric + QC columns
        measure_prefix_ol = st.text_input("Enter prefix for numeric measures (e.g., MRORRES)", "MRORRES", key="ol_prefix")
        mrorres_cols = [c for c in df_ol.columns if str(c).upper().startswith(measure_prefix_ol.upper())]
        numeric_cols, num_df = [], {visit_col: df_ol[visit_col]}
        for c in mrorres_cols: # Re-use mrorres_cols to be consistent with original intent
            col_num = pd.to_numeric(df_ol[c], errors="coerce")
            if col_num.notna().sum() > 1:
                numeric_cols.append(c)
                num_df[c] = col_num
        num_df = pd.DataFrame(num_df)

        qc_cols = [c for c in mrorres_cols
                if set(df_ol[c].dropna().unique()).issubset({"pass","fail"}) and df_ol[c].notna().any()]

        # Required global gate
        default_target_qc_ol = "MRORRES IMVOLQC" if "MRORRES IMVOLQC" in qc_cols else (qc_cols[0] if qc_cols else None)
        target_col = st.selectbox("Select Primary QC Gate for Outlier Detection", qc_cols, index=qc_cols.index(default_target_qc_ol) if default_target_qc_ol else 0, key="outlier_qc_gate")

        if target_col not in qc_cols:
            st.error(f"Required QC gate column '{target_col}' not found or not pass/fail.")
            st.stop()

        # Simplified QC mapping
        def qc_for_measure(meas_col: str) -> str | None:
            for suffix in ["QCC", "QC", "_QC"]:
                potential_qc_col = f"{meas_col}{suffix}"
                if potential_qc_col in qc_cols:
                    return potential_qc_col
            base_name = str(meas_col).replace(measure_prefix_ol, "").strip()
            for qc_col in qc_cols:
                if base_name in qc_col:
                    return qc_col
            return None

        num2qc = {c: qc_for_measure(c) for c in numeric_cols}
        numeric_cols = [c for c in numeric_cols if num2qc[c] is not None]

        # Visit order from sidebar
        visit_order = selected_visits

        # ---- Outlier computation (>3 SD), gated by IMVOLQC + per-measure QC
        outlier_records = []
        stats_records = []

        for meas in numeric_cols:
            friendly_name = code_to_label.get(extract_code(meas), meas)
            qcol_meas = num2qc[meas]

            for visit in visit_order:
                mask = (
                    (df_ol[visit_col] == visit) &
                    (df_ol[target_col] == "pass") &
                    (df_ol[qcol_meas] == "pass")
                )
                if not mask.any():
                    continue

                vals = num_df.loc[mask, [meas]].copy()
                subj_ids = df_ol.loc[mask, subjid_col].values

                if vals.empty:
                    continue

                mean = vals[meas].mean()
                sd   = vals[meas].std(ddof=1)
                # --- Guard against invalid SD ---
                if not np.isfinite(sd) or sd == 0:
                    lower, upper = np.nan, np.nan
                    outlier_mask = pd.Series(False, index=vals.index)
                else:
                    lower, upper = mean - 3 * sd, mean + 3 * sd
                    outlier_mask = (vals[meas] < lower) | (vals[meas] > upper)


                # summary row
                stats_records.append({
                    "VISIT": visit,
                    "Measure": friendly_name,
                    "Mean": mean,
                    "SD": sd,
                    "Lower_Limit": lower,
                    "Upper_Limit": upper,
                    "N": int(vals[meas].notna().sum()),
                    "Outlier_Count": int(outlier_mask.sum())
                })

                # detailed outliers
                if outlier_mask.any():
                    outliers = vals.loc[outlier_mask, meas]
                    subj_out = np.array(subj_ids)[outlier_mask.values]
                    for s_id, val in zip(subj_out, outliers.values):
                        outlier_records.append({
                            "VISIT": visit,
                            "Measure": friendly_name,
                            "SUBJID": s_id, # Keep column name SUBJID for output consistency
                            "Value": val,
                            "Mean": mean,
                            "SD": sd,
                            "Lower_Limit": lower,
                            "Upper_Limit": upper
                        })

        outliers_df = pd.DataFrame(outlier_records)
        summary_df  = pd.DataFrame(stats_records)

        # Round numeric columns
        for df__ in [outliers_df, summary_df]:
            for c in ["Mean","SD","Lower_Limit","Upper_Limit","Value"]:
                if c in df__.columns:
                    df__[c] = df__[c].round(3)

        # Show in app
        st.markdown("**Summary (per measure/visit)**")
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("**Outlier details (subject-level)**")
        if outliers_df.empty:
            st.info("No outliers detected (> 3 SD) under the QC-pass gating.")
        else:
            st.dataframe(outliers_df, use_container_width=True)

        # Download Excel with two sheets
        buf_out = io.BytesIO()
        with pd.ExcelWriter(buf_out, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Outlier_Summary_Stats", index=False)
            outliers_df.to_excel(writer, sheet_name="Outlier_Details", index=False)

        st.download_button(
            label="Download Outlier Results (Excel)",
            data=buf_out.getvalue(),
            file_name="Outliers_QCpass_3SD.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
   
    # -------------------------------
    #  Plots by visit (QC-pass only) + dots (friendly titles)
    # -------------------------------
    st.subheader("🚩 Plot Box + Dots (QC-pass only)")
    
    import re
    import os
    import zipfile

    if st.button("Generate Plots"):
        df_plot = df.copy()
        df_plot = normalize_pass_fail_columns(df_plot, visit_col)

        for c in df_plot.select_dtypes(include=["object","string","category"]).columns:
            if c != "VISIT":
                df_plot[c] = normalize_pf(df_plot[c])

        # ---- Friendly labels from MRITEST ----
        def extract_code(name: str) -> str:
            return re.sub(r'^\s*(MRORRES|MRITEST)\s*', '', str(name), flags=re.I).strip().upper()

        code_to_label = {}
        for col in [c for c in df_plot.columns if str(c).upper().startswith("MRITEST")]:
            ser = df_plot[col].dropna().astype(str).str.strip()
            if not ser.empty:
                mode = ser[ser != ""].mode()
                code_to_label[extract_code(col)] = (mode.iat[0] if not mode.empty else ser.iloc[0])

        def pretty(meas_col: str) -> str:
            return code_to_label.get(extract_code(meas_col), meas_col)

        # ---- MRORRES numeric measures & QC mapping ----
        measure_prefix_plot = st.text_input("Enter prefix for numeric measures (e.g., MRORRES)", "MRORRES", key="plot_prefix")
        mrorres_cols = [c for c in df_plot.columns if str(c).upper().startswith(measure_prefix_plot.upper())]
        numeric_cols, num_df = [], {visit_col: df_plot[visit_col]}
        for c in mrorres_cols:
            s_ = pd.to_numeric(df_plot[c], errors="coerce")
            if s_.notna().sum() > 1:
                numeric_cols.append(c); num_df[c] = s_
        num_df = pd.DataFrame(num_df)

        qc_cols = [c for c in mrorres_cols
                if set(df_plot[c].dropna().unique()).issubset({"pass","fail"}) and df_plot[c].notna().any()]

        def qc_for_measure(meas_col: str) -> str | None:
            for suffix in ["QCC", "QC", "_QC"]:
                potential_qc_col = f"{meas_col}{suffix}"
                if potential_qc_col in qc_cols:
                    return potential_qc_col
            base_name = str(meas_col).replace(measure_prefix_plot, "").strip()
            for qc_col in qc_cols:
                if base_name in qc_col:
                    return qc_col
            return None

        num2qc = {c: qc_for_measure(c) for c in numeric_cols}
        numeric_cols = [c for c in numeric_cols if num2qc[c] is not None]

        # Use visits from sidebar
        visit_order = selected_visits

        # ---- Build long table of QC-pass rows only ----
        def build_long_qcpass(df_in, num_df_in, measures, num2qc_map):
            recs = []
            for meas in measures:
                qcol = num2qc_map[meas]
                if qcol is None: continue
                mask = (df_in[qcol] == "pass") & num_df_in[meas].notna() & df_in[visit_col].notna()
                if mask.any():
                    recs.append(pd.DataFrame({
                        "Measure": meas,
                        "VISIT": df_in.loc[mask, visit_col].values,
                        "Value": num_df_in.loc[mask, meas].values
                    }))
            return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=["Measure","VISIT","Value"])

        long_df = build_long_qcpass(df_plot, num_df, numeric_cols, num2qc)
        long_df["VISIT"] = pd.Categorical(long_df["VISIT"], categories=visit_order, ordered=True)

        # ---- UI controls ----
        if long_df.empty:
            st.info("No QC-passed numeric MRORRES measures found to plot.")
        else:
            measures = sorted(long_df["Measure"].unique())
            left, right = st.columns([1.2,1])
            with left:
                selected_measures = st.multiselect("Select measures to plot",
                                                options=measures,
                                                default=measures[:min(6, len(measures))],
                                                format_func=lambda m: pretty(m))

            c1, c2, c3 = st.columns(3)
            with c1:
                jitter = st.slider("Jitter", 0.0, 0.3, 0.08, 0.01)
            with c2:
                point_alpha = st.slider("Dot alpha", 0.1, 1.0, 0.45, 0.05)
            with c3:
                point_size = st.slider("Dot size", 6, 40, 18, 1)
            dpi = st.slider("Save DPI", 80, 600, 300, 10)

            # ---- Internal helpers for plotting ----
            def _prepare_groups(long_df_in, measure, visits):
                subset = long_df_in[(long_df_in["Measure"] == measure) & (long_df_in["VISIT"].isin(visits))]
                grouped = subset.groupby("VISIT")["Value"].apply(list)
                if grouped.empty:
                    return [], []
                
                return grouped.index.tolist(), grouped.tolist()

            def plot_measure_box_with_dots(long_df_in, measure, visits, save_path=None,
                                        title_suffix=" ", figsize=(6,6)):
                labels_, data_ = _prepare_groups(long_df_in, measure, visits)
                if not data_:
                    return None

                fig = plt.figure(figsize=figsize)
                plt.boxplot(
                    data_,
                    widths=0.5,
                    showfliers=False,  # hide default outlier circles
                    medianprops=dict(color='black', linewidth=2.0),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.25),
                    capprops=dict(linewidth=1.25),
                    flierprops=dict(markersize=3)
                )

                rng = np.random.default_rng(42)
                y_min = min(d.min() for d in data_)
                y_max = max(d.max() for d in data_)
                span = (y_max - y_min) or 1.0
                for i, vals in enumerate(data_, start=1):
                    x = rng.normal(loc=i, scale=jitter, size=len(vals))
                    plt.scatter(x, vals, s=point_size, alpha=point_alpha, zorder=3)
                    # N label just below axis ticks (you can remove these two lines if you prefer corner N)
                    plt.text(i, y_min - 0.02 * span, f"N={len(vals)}",
                            ha="center", va="top", fontsize=12)

                nice = pretty(measure)
                plt.xticks(range(1, len(labels_) + 1), labels_, fontsize=13)

                # Conditional y-label
                if any("Screening" in v for v in labels_):
                    ylabel_text = "Volume (mm³)"
                elif any(v in ["Week52", "Week80"] for v in labels_):
                    ylabel_text = "Volume Atrophy Percent (%)"
                else:
                    ylabel_text = nice.title()
                plt.ylabel(ylabel_text, fontsize=13)

                title_text = f"{nice} {title_suffix}".title()
                plt.title(title_text, fontsize=16, pad=12)
                plt.grid(axis="y", linestyle="--", alpha=0.35)
                plt.tight_layout()

                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                return fig

            # ---- Generate plots and collect for ZIP ----
            generated_files = []
            for m in selected_measures:
                fig = plot_measure_box_with_dots(long_df, m, visit_order, title_suffix="")
                if fig is None:
                    continue
                st.pyplot(fig)
                # Save to memory for ZIP
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format="png", dpi=dpi, bbox_inches="tight")
                buf_png.seek(0)
                fname = f"{pretty(m).replace(' ','_')}_box_dots.png"
                generated_files.append((fname, buf_png.getvalue()))
                plt.close(fig)

            if generated_files:
                # Offer ZIP download
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fname, data_bytes in generated_files:
                        zf.writestr(fname, data_bytes)
                st.download_button(
                    label=f"Download {len(generated_files)} plot(s) as ZIP",
                    data=zip_buf.getvalue(),
                    file_name="plots_qcpass.zip",
                    mime="application/zip"
                )
            else:
                st.info("No plots generated with the current selections.")
