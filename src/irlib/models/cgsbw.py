from infre.helpers.functions import cluster_graph, prune_graph
from infre.models import ConGSB, GSBWindow


# IMPORTANT: due to __mro__, ConGSBWindow searches ConGSB methods first
# a.k.a, ConGSBWindow leverages methods from ConGSB and not GSBWindow
class ConGSBWindow(ConGSB, GSBWindow):
    """
    Contextual Graphical Set-Based Window (ConGSBWindow) Information Retrieval Model.
    This model combines the features of the GSBWindow and the ConGSB model, by introducing
    a windowed approach on the collection's terms and leveraging methods from ConGSB.

    Parameters:
    -----------
    collection : object
        The collection over which IR tasks will be performed.

    window : int or float
        The size of the window to consider while processing terms.
        - If an integer, it represents the number of terms.
        - If a float, it represents the proportion of terms in a document.

    clusters : int
        Number of clusters to be used for the union graph.

    cond : dict, optional (default={})
        Pruning conditions for the graph. Can specify conditions in the form {'edge': value} or {'sim': value}.

    Attributes:
    -----------
    model : str
        Name of the model.

    labels : ndarray
        Cluster labels for the nodes of the graph.

    embeddings : ndarray
        Embeddings corresponding to the nodes of the graph.

    graph : object
        The pruned union graph based on the windowed approach.

    prune_percentage : float
        Percentage of the graph that has been pruned.

    Methods:
    --------
    _model() -> str :
        Returns the class name of the model.

    """

    def __init__(self, collection, window, clusters, cond={}):
        GSBWindow.__init__(self, collection, window)

        # model name
        self.model = self._model()

        # Cluster the graph and get labels and embeddings
        self.labels, self.embeddings = cluster_graph(self.graph, collection, clusters)

        # Prune the graph
        self.graph, self.prune_percentage = prune_graph(
            self.graph, collection, self.labels, self.embeddings, cond
        )

        # NW Weight of GSBs
        self._nwk()

        self._cnwk()

    def _model(self) -> str:
        return __class__.__name__
