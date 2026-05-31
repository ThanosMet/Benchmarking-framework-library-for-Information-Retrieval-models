# src/irlib/api/registry.py
"""
Model Registry — ο κατάλογος όλων των διαθέσιμων μοντέλων.

Για να προσθέσεις νέο μοντέλο:
    1. Βάλε το αρχείο στο src/irlib/models/
    2. Πρόσθεσε μια γραμμή εδώ στο REGISTRY
    Τίποτα άλλο δεν αλλάζει — το API το βλέπει αυτόματα.
"""

from models.GSB import GSBModel
from models.BM25 import BM25Model
from models.GoW import Gow
from models.WindowedGSB import WindowedGSBModel
from models.gsbw import GSBWindow
from models.pgsb import PGSB
from models.pgsbw import PGSBW
from models.cgsb import ConGSB
from models.cgsbw import ConGSBWindow

REGISTRY = {
    "GSB":         GSBModel,
    "BM25":        BM25Model,
    "GOW":         Gow,
    "WINDOWEDGSB": WindowedGSBModel,
    "GSBWINDOW":   GSBWindow,
    "PGSB":        PGSB,
    "PGSBW":       PGSBW,
    "CONGSB":      ConGSB,
    "CONGSBW":     ConGSBWindow
}


def get_model_class(name: str):
    """Επιστρέφει την κλάση του μοντέλου ή κάνει raise αν δεν υπάρχει."""
    if name not in REGISTRY:
        raise KeyError(f"Μοντέλο '{name}' δεν βρέθηκε. Διαθέσιμα: {list(REGISTRY.keys())}")
    return REGISTRY[name]


def list_models():
    return list(REGISTRY.keys())
