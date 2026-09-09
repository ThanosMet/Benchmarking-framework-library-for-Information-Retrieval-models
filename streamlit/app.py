# streamlit/app.py
"""
IR Benchmarking Framework — Streamlit UI

Start:
    streamlit run streamlit/app.py

Requires Flask API running at http://127.0.0.1:5000
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:5000"


# ---------------------------------------------------------------------------
# Helpers — API calls
# ---------------------------------------------------------------------------

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Flask API not found. Make sure it is running on port 5000.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Flask API not found. Make sure it is running on port 5000.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------

def page_home():
    st.title("📚 IR Model Benchmarking Framework")
    st.markdown("Comparison of information retrieval models on standard IR collections.")

    st.divider()

    # API status
    data = api_get("/models")
    if data:
        col1, col2 = st.columns(2)

        with col1:
            st.success("✅ Flask API — Online")
            models = data.get("models", [])
            st.metric("Available Models", len(models))
            st.write(", ".join(models))

        with col2:
            col_data = api_get("/collections")
            if col_data:
                collections = col_data.get("collections", [])
                st.metric("Available Collections", len(collections))
                st.write(", ".join(collections))

    st.divider()
    st.markdown("""
    ### Usage Instructions
    - **Run Model** — Select a model, collection, and parameters, and run the benchmark
    - **Results** — View precision/recall results per query
    - **Compare** — Compare multiple models with charts
    """)


# ---------------------------------------------------------------------------
# Page: Run Model
# ---------------------------------------------------------------------------

def page_run():
    st.title("⚙️ Run Model")

    # Load options from API
    models_data = api_get("/models")
    collections_data = api_get("/collections")
    params_data = api_get("/model_params")

    if not models_data or not collections_data:
        return

    models = models_data.get("models", [])
    collections = collections_data.get("collections", [])
    model_params = params_data or {}

    # --- 1. OPTIONS OUTSIDE THE FORM (For instant UI refresh) ---
    col_m, col_c = st.columns(2)
    with col_m:
        model = st.selectbox("Model", models)
    with col_c:
        collection = st.selectbox("Collection", collections)

    # --- 2. PARAMETERS FORM ---
    with st.form("run_form"):
        col1, col2 = st.columns(2)

        with col1:
            runs = st.number_input("Number of runs", min_value=1, max_value=10, value=1)
            k = st.number_input("Cutoff k (0 = all docs)", min_value=0, value=0)
        with col2:
            stopwords = st.checkbox("Stopwords", value=True)
            min_freq = st.number_input("Min frequency (apriori)", min_value=1, value=1)
            save = st.checkbox("Save to MongoDB", value=False)

        # --- Dynamic parameters per model ---
        extra_params = {}
        extra_fields = model_params.get(model, [])
        if extra_fields:
            st.divider()
            st.markdown(f"**Parameters {model}**")
            for field in extra_fields:
                if field.get("type") == "string":
                    val = st.text_input(
                        f"{field['name']} — {field['help']}",
                        value=str(field["default"]),
                    )
                else:
                    val = st.number_input(
                        f"{field['name']} — {field['help']}",
                        value=float(field["default"]),
                    )
                extra_params[field["name"]] = val

        submitted = st.form_submit_button("▶️ Run", type="primary")

    if submitted:
        payload = {
            "model": model,
            "collection": collection,
            "runs": runs,
            "k": k if k > 0 else None,
            "stopwords": stopwords,
            "min_freq": min_freq,
            "save": save,
            "params": extra_params,
        }

        with st.spinner(f"Running {model} on collection {collection}..."):
            result = api_post("/run", payload)

        if result:
            st.success(f"✅ Completed in {result['elapsed_sec']} sec")
            st.session_state["last_result"] = result
            st.session_state["last_model"] = model

            # --- Metrics ---
            col1, col2, col3 = st.columns(3)
            col1.metric("MAP (mean)", f"{result['map_mean']:.4f}")
            col2.metric("MAP (std)", f"{result['map_std']:.4f}")
            col3.metric("Time", f"{result['elapsed_sec']}s")

            # --- Precision/Recall per query (run_0) ---
            if result.get("precision"):
                precision_run0 = result["precision"][0]
                recall_run0 = result["recall"][0]

                df = pd.DataFrame({
                    "Query": range(1, len(precision_run0) + 1),
                    "Precision": precision_run0,
                    "Recall": recall_run0,
                })

                st.subheader("Precision & Recall per Query (Run 1)")
                fig = px.line(df, x="Query", y=["Precision", "Recall"],
                              title=f"{model} — {collection}",
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Data")
                st.dataframe(df, use_container_width=True)



# ---------------------------------------------------------------------------
# Page: Results
# ---------------------------------------------------------------------------

def page_results():
    st.title("📊 Saved Results")

    # =====================================================================
    # 1. "NEW PAGE" LOGIC: If 'View' was clicked, show the chart
    # =====================================================================
    if st.session_state.get("view_result_data"):
        r = st.session_state["view_result_data"]

        # Button to return to the table
        if st.button("⬅️ Back to Results List"):
            st.session_state.pop("view_result_data")
            st.rerun()

        st.divider()
        st.subheader(f"Analysis: {r.get('model')} — {r.get('collection')}")

        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("MAP (mean)", f"{r.get('map_mean', 0):.4f}")
        col2.metric("MAP (std)", f"{r.get('map_std', 0):.4f}")
        col3.metric("Time (sec)", f"{r.get('elapsed_sec', '')}s")

        if r.get("params"):
            with st.expander("⚙️ Execution Parameters"):
                st.json(r["params"])

        # Display Chart and Data Table (same as in Run)
        if r.get("precision") and r.get("recall") and len(r["precision"]) > 0:
            precision_run0 = r["precision"][0]
            recall_run0 = r["recall"][0]

            df = pd.DataFrame({
                "Query": range(1, len(precision_run0) + 1),
                "Precision": precision_run0,
                "Recall": recall_run0,
            })

            st.subheader(f"Precision & Recall per Query ({r.get('model')} — {r.get('collection')})")
            fig = px.line(df, x="Query", y=["Precision", "Recall"], markers=True)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📊 View & Download Raw Data"):
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"results_{r.get('model')}_{r.get('collection')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
        else:
            st.warning("⚠️ No saved Precision/Recall data for this record.")

        # Stops function execution here, to avoid drawing the table below!
        return

    # =====================================================================
    # 2. NORMAL VIEW (The Table)
    # =====================================================================
    col1, col2, col3 = st.columns(3)
    with col1:
        model_filter = st.text_input("Model filter", "")
    with col2:
        collection_filter = st.text_input("Collection filter", "")
    with col3:
        limit = st.number_input("Max results", min_value=1, max_value=100, value=20)

    params = f"?limit={limit}"
    if model_filter:
        params += f"&model={model_filter.upper()}"
    if collection_filter:
        params += f"&collection={collection_filter.upper()}"

    data = api_get(f"/results{params}")
    if not data:
        return

    results = data.get("results", [])
    if not results:
        st.info("No saved results found. Run a model with 'Save to MongoDB'.")
        return

    st.metric("Total results", data["count"])
    st.write("")

    # 1. Create table headers
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([2, 1, 1, 1, 1, 1.5, 1, 1])
    c1.markdown("**Model**")
    c2.markdown("**Collection**")
    c3.markdown("**MAP**")
    c4.markdown("**Std**")
    c5.markdown("**Runs**")
    c6.markdown("**Time**")
    c7.markdown("**Analysis**")
    c8.markdown("**Action**")

    # Header divider line
    st.markdown("<hr style='margin: 0; border: 1px solid #666;'>", unsafe_allow_html=True)

    # 2. Fill rows
    for idx, r in enumerate(results):
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([2, 1, 1, 1, 1, 1.5, 1, 1])

        c1.write(r.get("model", ""))
        c2.write(r.get("collection", ""))
        c3.write(f"{r.get('map_mean', 0):.4f}")
        c4.write(f"{r.get('map_std', 0):.4f}")
        c5.write(str(r.get("runs", "")))
        c6.write(f"{r.get('elapsed_sec', '')}s")

        # --- VIEW BUTTON ---
        if c7.button("View", key=f"view_{idx}"):
            st.session_state["view_result_data"] = r
            st.rerun()

        # --- DELETE BUTTON ---
        if c8.button("Delete", key=f"del_{idx}"):
            res = api_post("/results/delete", {"timestamp": r.get("timestamp")})
            if res:
                st.rerun()

        # Horizontal line (border) to cleanly separate rows
        st.markdown("<hr style='margin: 0; border: 0.5px solid #333;'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Compare
# ---------------------------------------------------------------------------

def page_compare():
    st.title("🔀 Compare Models")

    models_data = api_get("/models")
    collections_data = api_get("/collections")

    if not models_data or not collections_data:
        return

    models = models_data.get("models", [])
    collections = collections_data.get("collections", [])

    # --- 1. OPTIONS OUTSIDE THE FORM ---
    col_m, col_c = st.columns(2)
    with col_m:
        selected_models = st.multiselect("Models to compare", models, default=models[:2])
    with col_c:
        collection = st.selectbox("Collection", collections)

    # --- 2. PARAMETERS FORM ---
    with st.form("compare_form"):
        col1, col2 = st.columns(2)
        with col1:
            runs = st.number_input("Number of runs", min_value=1, max_value=5, value=1)
        with col2:
            k = st.number_input("Cutoff k (0 = all docs)", min_value=0, value=0)
            stopwords = st.checkbox("Stopwords", value=True)

        # --- Dynamic parameters for ALL selected models ---
        extra_params = {}
        model_params = api_get("/model_params") or {}

        needed_fields = {}
        for m in (selected_models or []):
            for field in model_params.get(m, []):
                needed_fields[field["name"]] = field

        if needed_fields:
            st.divider()
            st.markdown("**Additional Model Parameters**")
            for name, field in needed_fields.items():
                if field.get("type") == "string":
                    val = st.text_input(
                        f"{name} — {field['help']}",
                        value=str(field["default"]),
                    )
                else:
                    val = st.number_input(
                        f"{name} — {field['help']}",
                        value=float(field["default"]),
                    )
                extra_params[name] = val

        submitted = st.form_submit_button("▶️ Compare", type="primary")

    if submitted:
        if len(selected_models) < 2:
            st.warning("Select at least 2 models.")
            return

        payload = {
            "models": selected_models,
            "collection": collection,
            "runs": runs,
            "k": k if k > 0 else None,
            "stopwords": stopwords,
            "params": extra_params,
        }

        with st.spinner(f"Running models {', '.join(selected_models)}..."):
            data = api_post("/compare", payload)

        if not data:
            return

        results = data.get("results", {})
        errors = data.get("errors") or {}

        if errors:
            for m, err in errors.items():
                st.error(f"{m}: {err}")

        if not results:
            return

        # --- MAP comparison bar chart ---
        map_data = {m: r["map_mean"] for m, r in results.items()}
        time_data = {m: r["elapsed_sec"] for m, r in results.items()}

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("MAP Comparison")
            fig = px.bar(
                x=list(map_data.keys()),
                y=list(map_data.values()),
                labels={"x": "Model", "y": "MAP"},
                color=list(map_data.keys()),
                text=[f"{v:.4f}" for v in map_data.values()],
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Execution Time (sec)")
            fig2 = px.bar(
                x=list(time_data.keys()),
                y=list(time_data.values()),
                labels={"x": "Model", "y": "Seconds"},
                color=list(time_data.keys()),
                text=[f"{v:.2f}s" for v in time_data.values()],
            )
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

        # --- Precision per query overlay ---
        st.subheader("Precision per Query")
        fig3 = go.Figure()
        for m, r in results.items():
            if r.get("precision"):
                precision_run0 = r["precision"][0]
                fig3.add_trace(go.Scatter(
                    x=list(range(1, len(precision_run0) + 1)),
                    y=precision_run0,
                    name=m,
                    mode="lines+markers",
                ))
        fig3.update_layout(xaxis_title="Query", yaxis_title="Precision")
        st.plotly_chart(fig3, use_container_width=True)

        # --- Summary table ---
        st.subheader("Summary")
        rows = []
        for m, r in results.items():
            rows.append({
                "Model": m,
                "MAP": round(r["map_mean"], 4),
                "Std": round(r["map_std"], 4),
                "Time (s)": r["elapsed_sec"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="IR Benchmarking",
        page_icon="📚",
        layout="wide",
    )

    pages = {
        "🏠 Home": page_home,
        "⚙️ Run Model": page_run,
        "📊 Results": page_results,
        "🔀 Compare": page_compare,
    }

    with st.sidebar:
        st.title("IR Benchmarking")
        st.markdown("---")
        selected = st.radio("Navigation", list(pages.keys()))

    if selected != "📊 Results" and "view_result_data" in st.session_state:
        del st.session_state["view_result_data"]

    pages[selected]()


if __name__ == "__main__":
    main()