from models.GSB import GSBModel as GSB
from models.Model import Model as BaseIRModel
from utilities.functions import cluster_graph, prune_graph

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
    
    condition : dict, default={}
        Pruning conditions. Can be {'edge': value} or {'sim': value}.
    """

    
    def __init__(self, collection, clusters, condition={}):
        """Initialize the PGSB model with the given collection, clusters, and pruning conditions."""

        # 1. Initialize the parent GSB class
        # (This automatically sets up k_core_bool AND builds self.graph!)
        super().__init__(collection)
        
        # model name
        self.model = self.__class__.__name__

        # Cluster the graph and get labels and embeddings
        self.labels, self.embeddings = cluster_graph(self.graph, collection, clusters)

        # Prune the graph
        self.graph, self.prune_percentage = prune_graph(self.graph, collection, self.labels, self.embeddings, condition)

    def _model(self): 
        return __class__.__name__

