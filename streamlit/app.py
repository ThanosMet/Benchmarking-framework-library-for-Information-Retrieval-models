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
    params_data = api_get("/model_params")

    if not models_data or not collections_data:
        return

    models = models_data.get("models", [])
    collections = collections_data.get("collections", [])
    model_params = params_data or {}

    # --- 1. ΕΠΙΛΟΓΕΣ ΕΞΩ ΑΠΟ ΤΗ ΦΟΡΜΑ (Για άμεση ανανέωση του UI) ---
    col_m, col_c = st.columns(2)
    with col_m:
        model = st.selectbox("Μοντέλο", models)
    with col_c:
        collection = st.selectbox("Συλλογή", collections)

    # --- 2. ΦΟΡΜΑ ΠΑΡΑΜΕΤΡΩΝ ---
    with st.form("run_form"):
        col1, col2 = st.columns(2)

        with col1:
            runs = st.number_input("Αριθμός runs", min_value=1, max_value=10, value=1)
            k = st.number_input("Cutoff k (0 = όλα τα docs)", min_value=0, value=0)
        with col2:
            stopwords = st.checkbox("Stopwords", value=True)
            min_freq = st.number_input("Min frequency (apriori)", min_value=1, value=1)
            save = st.checkbox("Αποθήκευση στη MongoDB", value=False)

        # --- Δυναμικές παράμετροι ανά μοντέλο ---
        extra_params = {}
        extra_fields = model_params.get(model, [])
        if extra_fields:
            st.divider()
            st.markdown(f"**Παράμετροι {model}**")
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

    # =====================================================================
    # 1. ΛΟΓΙΚΗ "ΝΕΑΣ ΣΕΛΙΔΑΣ": Αν έχουμε πατήσει 'View', δείχνουμε το γράφημα
    # =====================================================================
    if st.session_state.get("view_result_data"):
        r = st.session_state["view_result_data"]

        # Κουμπί για επιστροφή στον πίνακα
        if st.button("⬅️ Επιστροφή στη Λίστα Αποτελεσμάτων"):
            st.session_state.pop("view_result_data")
            st.rerun()

        st.divider()
        st.subheader(f"Ανάλυση: {r.get('model')} — {r.get('collection')}")

        # Εμφάνιση των Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("MAP (mean)", f"{r.get('map_mean', 0):.4f}")
        col2.metric("MAP (std)", f"{r.get('map_std', 0):.4f}")
        col3.metric("Χρόνος (sec)", f"{r.get('elapsed_sec', '')}s")

        if r.get("params"):
            with st.expander("⚙️ Παράμετροι Εκτέλεσης"):
                st.json(r["params"])

        # Εμφάνιση του Γραφήματος και του Πίνακα Δεδομένων (όπως στο Run)
        if r.get("precision") and r.get("recall") and len(r["precision"]) > 0:
            precision_run0 = r["precision"][0]
            recall_run0 = r["recall"][0]

            df = pd.DataFrame({
                "Query": range(1, len(precision_run0) + 1),
                "Precision": precision_run0,
                "Recall": recall_run0,
            })

            st.subheader(f"Precision & Recall ανά Query ({r.get('model')} — {r.get('collection')})")
            fig = px.line(df, x="Query", y=["Precision", "Recall"], markers=True)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📊 Προβολή & Λήψη Αναλυτικών Δεδομένων (Raw Data)"):
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Λήψη σε CSV",
                    data=csv,
                    file_name=f"results_{r.get('model')}_{r.get('collection')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )
        else:
            st.warning("⚠️ Δεν υπάρχουν αποθηκευμένα δεδομένα Precision/Recall για αυτή την εγγραφή.")

        # Σταματάει την εκτέλεση της συνάρτησης εδώ, για να μην σχεδιάσει τον πίνακα από κάτω!
        return

        # =====================================================================
    # 2. ΚΑΝΟΝΙΚΗ ΠΡΟΒΟΛΗ (Ο Πίνακας)
    # =====================================================================
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
    st.write("")

    # 1. Φτιάχνουμε τις κεφαλίδες του πίνακα
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([2, 1, 1, 1, 1, 1.5, 1, 1])
    c1.markdown("**Μοντέλο**")
    c2.markdown("**Συλλογή**")
    c3.markdown("**MAP**")
    c4.markdown("**Std**")
    c5.markdown("**Runs**")
    c6.markdown("**Χρόνος**")
    c7.markdown("**Ανάλυση**")
    c8.markdown("**Ενέργεια**")

    # Διαχωριστική γραμμή κεφαλίδας
    st.markdown("<hr style='margin: 0; border: 1px solid #666;'>", unsafe_allow_html=True)

    # 2. Γεμίζουμε τις σειρές
    for idx, r in enumerate(results):
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([2, 1, 1, 1, 1, 1.5, 1, 1])

        c1.write(r.get("model", ""))
        c2.write(r.get("collection", ""))
        c3.write(f"{r.get('map_mean', 0):.4f}")
        c4.write(f"{r.get('map_std', 0):.4f}")
        c5.write(str(r.get("runs", "")))
        c6.write(f"{r.get('elapsed_sec', '')}s")

        # --- ΤΟ ΚΟΥΜΠΙ VIEW ---
        if c7.button("View", key=f"view_{idx}"):
            st.session_state["view_result_data"] = r
            st.rerun()

        # --- ΤΟ ΚΟΥΜΠΙ DELETE ---
        if c8.button("Delete", key=f"del_{idx}"):
            res = api_post("/results/delete", {"timestamp": r.get("timestamp")})
            if res:
                st.rerun()

        # Οριζόντια γραμμή (border) για να χωρίζει όμορφα τις σειρές
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

    # --- 1. ΕΠΙΛΟΓΕΣ ΕΞΩ ΑΠΟ ΤΗ ΦΟΡΜΑ ---
    col_m, col_c = st.columns(2)
    with col_m:
        selected_models = st.multiselect("Μοντέλα προς σύγκριση", models, default=models[:2])
    with col_c:
        collection = st.selectbox("Συλλογή", collections)

    # --- 2. ΦΟΡΜΑ ΠΑΡΑΜΕΤΡΩΝ ---
    with st.form("compare_form"):
        col1, col2 = st.columns(2)
        with col1:
            runs = st.number_input("Αριθμός runs", min_value=1, max_value=5, value=1)
        with col2:
            k = st.number_input("Cutoff k (0 = όλα τα docs)", min_value=0, value=0)
            stopwords = st.checkbox("Stopwords", value=True)

        # --- Δυναμικές παράμετροι για ΟΛΑ τα επιλεγμένα μοντέλα ---
        extra_params = {}
        model_params = api_get("/model_params") or {}

        needed_fields = {}
        for m in (selected_models or []):
            for field in model_params.get(m, []):
                needed_fields[field["name"]] = field

        if needed_fields:
            st.divider()
            st.markdown("**Επιπλέον Παράμετροι Μοντέλων**")
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
            st.warning("Επίλεξε τουλάχιστον 2 μοντέλα.")
            return

        payload = {
            "models": selected_models,
            "collection": collection,
            "runs": runs,
            "k": k if k > 0 else None,
            "stopwords": stopwords,
            "params": extra_params,
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

    if selected != "📊 Results" and "view_result_data" in st.session_state:
        del st.session_state["view_result_data"]

    pages[selected]()


if __name__ == "__main__":
    main()
