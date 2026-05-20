from numpy import zeros
from infre import metrics
from infre.models import GSB

class GSBWindow(GSB):
    """
    This is an extension of the GSB (Graphical Set Based) model which incorporates the idea of 
    windowing for adjacency matrix computation. Rather than considering the entire document at once, 
    it looks at smaller 'windows' or segments of the document.

    Parameters:
    -----------
    collection : object
        The document collection on which the GSB model is to be built.
    
    window : int or float, default=7
        Size of the window to consider while computing the adjacency matrix.
        - If given as an int, it's considered as the fixed number of terms.
        - If given as a float, it's considered as a percentage of the total terms in the document.
    """
        
    def __init__(self, collection, window=7):
        """Initialize the GSBWindow model with the given collection and window size."""
        self.window = window
        super().__init__(collection)


    def class_name(self): return __class__.__name__


    def doc2adj(self, document):
        """
        Overridden method to compute the adjacency matrix based on the windowed 
        segments of the document.
        
        Parameters:
        -----------
        document : object
            The document object which contains terms and their term frequencies.

        Returns:
        --------
        ndarray
            The adjacency matrix for the given document based on windowed term-term relations.
        """

        window_size = - 1
        if isinstance(self.window, int):
            window_size = self.window
        elif isinstance(self.window, float):
            window_size = int(self.window * len(document.terms))

        # create windowed document
        windowed_doc = document.split_document(window_size)

        adj_matrix = zeros(shape=(len(document.tf), len(document.tf)), dtype=int)
        for segment in windowed_doc:
            w_tf = metrics.tf(segment)

            for i, term_i in enumerate(document.tf):
                for j, term_j in enumerate(document.tf):
                    if term_i in w_tf.keys() and term_j in w_tf.keys():
                        if i == j:
                            adj_matrix[i][j] += w_tf[term_i] * (w_tf[term_i] + 1) / 2
                        else:
                            adj_matrix[i][j] += w_tf[term_i] * w_tf[term_j]
        return adj_matrix