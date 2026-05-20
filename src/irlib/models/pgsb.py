from infre.models import GSB, BaseIRModel
from infre.helpers.functions import cluster_graph, prune_graph

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
        BaseIRModel.__init__(self, collection)
        
        # model name
        self.model = self._model()

        # create Union graph
        self.graph = self.union_graph()

        # Cluster the graph and get labels and embeddings
        self.labels, self.embeddings = cluster_graph(self.graph, collection, clusters)

        # Prune the graph
        self.graph, self.prune_percentage = prune_graph(self.graph, collection, self.labels, self.embeddings, condition)
        
        # NW Weight of GSBs
        self._nwk()


    def _model(self): 
        return __class__.__name__

