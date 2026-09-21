# IR Benchmarking Framework

An extensible benchmarking framework for the execution and comparative
evaluation of heterogeneous **Information Retrieval (IR) models** under
a common experimental pipeline.

This project was developed as part of my Diploma Thesis at the Department of Computer Engineering and Informatics (CEID), 
University of Patras.

## Overview

Information Retrieval models can differ significantly in their
representation, parameterization and retrieval mechanisms.

The goal of this project is not to introduce a new retrieval model,
but to provide a common environment where different IR approaches can
be integrated, configured, executed and compared under the same
experimental conditions.

The framework provides:

- a common interface for IR models,
- a central Model Registry,
- unified collection loading,
- dynamic model parameterization,
- single-model experiments,
- multi-model comparison,
- evaluation using common IR metrics,
- execution-time measurements,
- MongoDB result persistence,
- CSV result export,
- a Streamlit graphical interface,
- a Flask REST API.

---

## Supported Models

The current version includes twelve models and variants:

### Classical / Lexical

- TF-IDF
- LSI
- BM25

### Graph-based

- Graph of Words (GoW)
- Graphical Set-Based (GSB)
- WindowedGSB
- Pruned Graphical Set-Based (PGSB)
- PGSBW
- Contextual Graphical Set-Based (ConGSB)
- ConGSBW

### Neural

- Sentence-BERT (SBERT)
- PyLate / ColBERT

---

## Test Collections

The framework currently supports three standard Information Retrieval
test collections:

- **Cystic Fibrosis (CF)**
- **Cranfield (CRAN)**
- **National Physical Laboratory (NPL)**

Each collection contains:

- documents,
- queries,
- relevance judgments (qrels).

Collections are converted into a common internal representation before
being passed to the retrieval models.

---

## Evaluation Metrics

The main evaluation metrics are:

- **Mean Average Precision (MAP)**
- **Precision@10**
- **Recall@10**
- **Execution Time**

Average Precision and MAP are calculated using the complete document
ranking, while Precision@10 and Recall@10 evaluate the first ten
retrieved documents.

For models involving stochastic procedures, multiple runs can be
executed and the framework reports the mean MAP and standard deviation.

---

## Architecture

The application follows a layered architecture:

```text
User
 │
 ▼
Streamlit Frontend
 │
 │ HTTP / JSON
 ▼
Flask REST API
 │
 ├──────────────► Model Registry
 │                    │
 │                    ▼
 │                 IR Models
 │
 └──────────────► Collection Builder
                      │
                      ▼
                   MongoDB
```

### Main Components

- **Streamlit Frontend**  
  Experiment configuration, execution and visualization.

- **Flask REST API**  
  Handles requests and coordinates experiment execution.

- **Model Registry**  
  Maps model identifiers to their corresponding Python classes.

- **Collection Builder**  
  Converts data stored in MongoDB into the common collection
  representation used by the models.

- **MongoDB**  
  Stores test collections and optionally experimental results.

---

## Project Structure

```text
ir-model-comparison/
│
├── collections/
│   ├── CF/
│   ├── CRAN/
│   └── NPL/
│
├── scripts/
│   ├── insert_cf_collection.py
│   ├── insert_cran_collection.py
│   └── insert_npl_collection.py
│
├── src/
│   └── irlib/
│       │
│       ├── api/
│       │   ├── app.py
│       │   └── registry.py
│       │
│       ├── datasets_insert/
│       │   ├── mongo_ingest.py
│       │   └── mongo_loader.py
│       │
│       ├── models/
│       │   ├── Model.py
│       │   ├── BM25.py
│       │   ├── GoW.py
│       │   ├── GSB.py
│       │   ├── WindowedGSB.py
│       │   ├── pgsb.py
│       │   ├── pgsbw.py
│       │   ├── cgsb.py
│       │   ├── cgsbw.py
│       │   ├── tfidf_model.py
│       │   ├── lsi.py
│       │   ├── sbert.py
│       │   └── pylate_colbert.py
│       │
│       ├── Preprocess/
│       │
│       ├── utilities/
│       │
│       └── collection_builder.py
│
├── streamlit/
│   └── app.py
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Model Registry

Models are registered centrally in the framework:

```python
REGISTRY = {
    "GSB": GSBModel,
    "BM25": BM25Model,
    "GOW": Gow,
    "WINDOWEDGSB": WindowedGSBModel,
    "PGSB": PGSB,
    "PGSBW": PGSBW,
    "CONGSB": ConGSB,
    "CONGSBW": ConGSBWindow,
    "PYLATE": PyLateColBERT,
    "TFIDF": TFIDFModel,
    "SBERT": SBERTModel,
    "LSI": LSIModel
}
```

This design makes it possible to extend the framework without creating
a separate evaluation pipeline for every new retrieval model.

---

## REST API

The Flask backend exposes the main functionality through the following
endpoints:

```text
/collections
/models
/model_params
/run
/compare
/results
```

The `/compare` endpoint allows multiple models to be executed on the
same collection under common experimental settings.

---

## User Interface

The Streamlit interface provides four main views:

### Home

Displays the availability of the backend, MongoDB connection and
general framework information.

### Run Model

Allows the user to:

- select a collection,
- select a retrieval model,
- configure model-specific parameters,
- select the number of runs,
- define the evaluation cutoff,
- optionally save results to MongoDB.

### Results

Displays:

- MAP,
- standard deviation,
- execution time,
- Precision and Recall per query,
- tabular results,
- CSV export.

### Compare Models

Multiple models can be executed on the same collection and compared
using:

- MAP,
- Precision@10,
- Recall@10,
- execution time.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ThanosMet/Benchmarking-framework-library-for-Information-Retrieval-models.git
cd Benchmarking-framework-library-for-Information-Retrieval-models
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install the dependencies:

```bash
pip install -r requirements.txt
```

MongoDB can be started using Docker Compose:

```bash
docker compose up -d
```

---

## Running the Application

Start the Flask backend:

```bash
python src/irlib/api/app.py
```

Then start the Streamlit frontend:

```bash
streamlit run streamlit/app.py
```

The Streamlit interface can then be used to configure and execute
experiments.

---

## Extending the Framework

A new retrieval model can be integrated by:

1. Implementing a model class compatible with the common model
   interface.
2. Adding the class to `src/irlib/models/`.
3. Registering the model in `registry.py`.
4. Defining any model-specific parameters required by the API.

Once registered, the model can use the same collection loading,
evaluation and comparison pipeline as the existing implementations.

---

## Technologies

- Python
- Flask
- Streamlit
- MongoDB
- Docker
- NumPy
- pandas
- scikit-learn
- NetworkX
- NLTK
- PyTorch
- Sentence-Transformers
- PyLate

---

## Diploma Thesis

This repository contains the software implementation developed for the
Diploma Thesis:

**Implementation of a Library for Comparing Information Retrieval Models**

Department of Computer Engineering and Informatics  
University of Patras  
2026

---

## Author

**Athanasios Metaxas**

---

## License

This repository was developed for academic and research purposes.