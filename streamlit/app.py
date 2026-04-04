# streamlit/app.py
"""
IR Benchmarking Framework — Streamlit UI

Εκκίνηση:
    streamlit run streamlit/app.py

Απαιτεί το Flask API να τρέχει στο http://127.0.0.1:5000
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
        st.error("❌ Δεν βρέθηκε το Flask API. Βεβαιώσου ότι τρέχει στο port 5000.")
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
        st.error("❌ Δεν βρέθηκε το Flask API. Βεβαιώσου ότι τρέχει στο port 5000.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------

def page_home():
    st.title("📚 IR Model Benchmarking Framework")
    st.markdown("Σύγκριση μοντέλων ανάκτησης πληροφορίας πάνω σε standard IR collections.")

    st.divider()

    # API status
    data = api_get("/models")
    if data:
        col1, col2 = st.columns(2)

        with col1:
            st.success("✅ Flask API — Online")
            models = data.get("models", [])
            st.metric("Διαθέσιμα μοντέλα", len(models))
            st.write(", ".join(models))

        with col2:
            col_data = api_get("/collections")
            if col_data:
                collections = col_data.get("collections", [])
                st.metric("Διαθέσιμες συλλογές", len(collections))
                st.write(", ".join(collections))

    st.divider()
    st.markdown("""
    ### Οδηγίες χρήσης
    - **Run Model** — Επίλεξε μοντέλο, συλλογή και παραμέτρους και τρέξε το benchmark
    - **Results** — Δες τα αποτελέσματα precision/recall ανά query
    - **Compare** — Σύγκριση πολλών μοντέλων με γραφήματα
    """)


# ---------------------------------------------------------------------------
# Page: Run Model
# ---------------------------------------------------------------------------

def page_run():
    st.title("⚙️ Run Model")

    # Φόρτωση επιλογών από API
    models_data = api_get("/models")
    collections_data = api_get("/collections")

    if not models_data or not collections_data:
        return

    models = models_data.get("models", [])
    collections = collections_data.get("collections", [])

    # --- Φόρμα παραμέτρων ---
    with st.form("run_form"):
        col1, col2 = st.columns(2)

        with col1:
            model = st.selectbox("Μοντέλο", models)
            collection = st.selectbox("Συλλογή", collections)

        with col2:
            runs = st.number_input("Αριθμός runs", min_value=1, max_value=10, value=1)
            k = st.number_input("Cutoff k (0 = όλα τα docs)", min_value=0, value=0)
            stopwords = st.checkbox("Stopwords", value=True)
            min_freq = st.number_input("Min frequency (apriori)", min_value=1, value=1)
            save = st.checkbox("Αποθήκευση στη MongoDB", value=False)

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
        }

        with st.spinner(f"Τρέχει το {model} στη συλλογή {collection}..."):
            result = api_post("/run", payload)

        if result:
            st.success(f"✅ Ολοκληρώθηκε σε {result['elapsed_sec']} sec")
            st.session_state["last_result"] = result
            st.session_state["last_model"] = model

            # --- Metrics ---
            col1, col2, col3 = st.columns(3)
            col1.metric("MAP (mean)", f"{result['map_mean']:.4f}")
            col2.metric("MAP (std)", f"{result['map_std']:.4f}")
            col3.metric("Χρόνος", f"{result['elapsed_sec']}s")

            # --- Precision/Recall ανά query (run_0) ---
            if result.get("precision"):
                precision_run0 = result["precision"][0]
                recall_run0 = result["recall"][0]

                df = pd.DataFrame({
                    "Query": range(1, len(precision_run0) + 1),
                    "Precision": precision_run0,
                    "Recall": recall_run0,
                })

                st.subheader("Precision & Recall ανά Query (Run 1)")
                fig = px.line(df, x="Query", y=["Precision", "Recall"],
                              title=f"{model} — {collection}",
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Δεδομένα")
                st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Results
# ---------------------------------------------------------------------------

def page_results():
    st.title("📊 Αποθηκευμένα Αποτελέσματα")

    col1, col2, col3 = st.columns(3)
    with col1:
        model_filter = st.text_input("Φίλτρο μοντέλου", "")
    with col2:
        collection_filter = st.text_input("Φίλτρο συλλογής", "")
    with col3:
        limit = st.number_input("Max αποτελέσματα", min_value=1, max_value=100, value=20)

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
        st.info("Δεν υπάρχουν αποθηκευμένα αποτελέσματα. Τρέξε ένα μοντέλο με 'Αποθήκευση στη MongoDB'.")
        return

    st.metric("Σύνολο αποτελεσμάτων", data["count"])

    # Summary table
    rows = []
    for r in results:
        rows.append({
            "Μοντέλο": r.get("model", ""),
            "Συλλογή": r.get("collection", ""),
            "MAP": round(r.get("map_mean", 0), 4),
            "Std": round(r.get("map_std", 0), 4),
            "Runs": r.get("runs", ""),
            "Χρόνος (s)": r.get("elapsed_sec", ""),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


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

    with st.form("compare_form"):
        selected_models = st.multiselect("Μοντέλα προς σύγκριση", models, default=models[:2])
        collection = st.selectbox("Συλλογή", collections)
        runs = st.number_input("Αριθμός runs", min_value=1, max_value=5, value=1)
        k = st.number_input("Cutoff k (0 = όλα τα docs)", min_value=0, value=0)
        stopwords = st.checkbox("Stopwords", value=True)

        submitted = st.form_submit_button("▶️ Compare", type="primary")

    if submitted:
        if len(selected_models) < 2:
            st.warning("Επίλεξε τουλάχιστον 2 μοντέλα.")
            return

        payload = {
            "models": selected_models,
            "collection": collection,
            "runs": runs,
            "k": k if k > 0 else None,
            "stopwords": stopwords,
        }

        with st.spinner(f"Τρέχουν τα μοντέλα {', '.join(selected_models)}..."):
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
            st.subheader("MAP Σύγκριση")
            fig = px.bar(
                x=list(map_data.keys()),
                y=list(map_data.values()),
                labels={"x": "Μοντέλο", "y": "MAP"},
                color=list(map_data.keys()),
                text=[f"{v:.4f}" for v in map_data.values()],
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Χρόνος Εκτέλεσης (sec)")
            fig2 = px.bar(
                x=list(time_data.keys()),
                y=list(time_data.values()),
                labels={"x": "Μοντέλο", "y": "Seconds"},
                color=list(time_data.keys()),
                text=[f"{v:.2f}s" for v in time_data.values()],
            )
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

        # --- Precision per query overlay ---
        st.subheader("Precision ανά Query")
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
        st.subheader("Σύνοψη")
        rows = []
        for m, r in results.items():
            rows.append({
                "Μοντέλο": m,
                "MAP": round(r["map_mean"], 4),
                "Std": round(r["map_std"], 4),
                "Χρόνος (s)": r["elapsed_sec"],
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

    pages[selected]()


if __name__ == "__main__":
    main()
