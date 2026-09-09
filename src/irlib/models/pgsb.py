from models.GSB import GSBModel as GSB
from models.Model import Model as BaseIRModel
from utilities.functions import cluster_graph, prune_graph
import ast


class PGSB(GSB, BaseIRModel):
    """
    Pruned Graphical Set Based (PGSB) Model.
    Extends GSB by clustering the union graph, then pruning it based on specified conditions.

    Parameters:
    -----------
    collection : object
        The document collection.

    clusters : int
        Number of clusters for the union graph.

    condition : dict or str, default={}
        Pruning conditions. Can be {'edge': value} or {'sim': value}.
    """

    def __init__(self, collection, clusters, condition={}):
        """Initialize the PGSB model with the given collection, clusters, and pruning conditions."""

        # 1. Initialize the parent GSB class
        super().__init__(collection)

        # model name
        self.model = self.__class__.__name__

        # --- FIX: Μετατροπή του condition από String σε Dictionary ---
        if isinstance(condition, str) and condition.strip() != "":
            try:
                condition = ast.literal_eval(condition)
            except Exception as e:
                print(f"Σφάλμα κατά τη μετατροπή του condition: {e}")
                condition = {}
        # -------------------------------------------------------------

        # Cluster the graph and get labels and embeddings
        self.labels, self.embeddings = cluster_graph(self.graph, collection, clusters)

        print(f"Ακμές ΠΡΙΝ το pruning (Clusters: {clusters}, Condition: {condition}):", self.graph.number_of_edges())

        # Prune the graph
        self.graph, self.prune_percentage = prune_graph(self.graph, collection, self.labels, self.embeddings, condition)

        print("Ακμές ΜΕΤΑ το pruning:", self.graph.number_of_edges())
        self._nwk = self._calculate_nwk()

    def _model(self):
        return __class__.__name__