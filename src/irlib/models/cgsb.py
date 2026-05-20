import numpy as np
from numpy import zeros
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from infre.helpers.functions import cluster_optimization, cluster_graph, prune_graph
from infre.metrics import precision_recall
from infre.models import GSB
from infre.tools import apriori


class ConGSB(GSB):
    """
    Contextual Graphical Set-Based (ConGSB) Information Retrieval Model.
    Extends the GSB model by introducing a contextual approach, performing clustering, and pruning.
    Also, incorporates query expansion and employs term-set based similarity calculations.

    Parameters:
    -----------
    collection : object
        The collection over which IR tasks will be performed.

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
        The pruned union graph.

    prune_percentage : float
        Percentage of the graph that has been pruned.

    Methods:
    --------
    _model() -> str :
        Returns the class name.

    expand_q(query: list, k: int) -> list :
        Expand the given query using nearest neighbor approach in the embeddings space.

    fit_evaluate(queries: list, relevant: list) -> tuple :
        Perform IR tasks on given queries and evaluate using the precision-recall metric.

    _cnwk() -> None :
        Calculate the new scalar centroids of NWk for each term in the collection's inverted index.

    _model_func(termsets: list) -> np.ndarray :
        Compute the model adjusting weight based on the contextual NWk values for each termset.
    """

    def __init__(self, collection, clusters, cond={}, **kwargs):
        super().__init__(collection)

        Valid_args = {"cluster_optimization"}
        for key in kwargs:
            if key not in Valid_args:
                raise ValueError(f"Invalid argument {key} provided.")

        self.cluster_optimization = kwargs.get("cluster_optimization", False)
        print(f"Cluster optimization is set to {self.cluster_optimization}")
        if self.cluster_optimization in {"eigen_gap", "elbow", "silhouette"}:
            self.clusters = cluster_optimization( graph = self.graph, collection = collection, method  = self.cluster_optimization)
            print(self.clusters)
        else:
            self.clusters = clusters

        # model name
        self.model = self._model()

        # Cluster the graph and get labels and embeddings
        self.labels, self.embeddings = cluster_graph(
            self.graph, collection, self.clusters
        )

        # Prune the graph
        self.graph, self.prune_percentage = prune_graph(
            self.graph, collection, self.labels, self.embeddings, cond
        )

        # calculate the new scalar centroids of NWk
        self._cnwk()



    def _model(self):
        return __class__.__name__

    def expand_q(self, query, k):

        inv_index = self.collection.inverted_index

        # Get collection indices for each term in query
        indices = [inv_index[term]["id"] for term in query if term in inv_index]

        # if query term not in collection, instantly append in expansion terms
        # Since there is no embedding calculate for it
        expansion_terms = [term for term in query if term not in inv_index]

        # Get query embeddings and drop label (cluster id)
        query_embeddings = self.embeddings.iloc[indices, :-1].values

        # Calculate query point in embedding space by taking the mean
        qv = np.mean(query_embeddings, axis=0)

        # Fit a k-nearest neighbors model on the collection embeddings
        knn_model = NearestNeighbors(n_neighbors=k, metric="cosine")
        knn_model.fit(self.embeddings.iloc[:, :-1].values)

        # Find the k-nearest neighbors to the query centroid
        _, top_k_indices = knn_model.kneighbors(np.array([qv]))

        # Convert the top_k_indices array to a 1D array
        top_k_indices = top_k_indices[0]

        col_terms = list(inv_index.keys())
        # Add the expansion terms to the list preventing duplicates
        expansion_terms.extend(
            [
                col_terms[k_ind]
                for k_ind in top_k_indices
                if col_terms[k_ind] not in query
            ]
        )

        return expansion_terms

    def expand_q_centroids(self, query, k):
        """
        Expands the given query by finding the most similar terms in the embedding space.
        Parameters:
        query (list of str): The list of query terms to be expanded.
        k (int): The number of expansion terms to be returned.
        Returns:
        list of str: A list of expanded query terms, including the original query terms and the most similar terms from the collection.
        The method performs the following steps:
        1. Retrieves the indices of the query terms from the inverted index.
        2. Identifies query terms that are not present in the collection and adds them to the expansion terms.
        3. Computes the embeddings for the query terms.
        4. Calculates the centroid of the query terms in the embedding space.
        5. Computes the centroids of each cluster in the embedding space.
        6. Determines the cluster assignment for the query centroid.
        7. Identifies terms from the collection that belong to the same cluster as the query terms.
        8. Calculates the cosine similarity between the query centroid and the terms in the same cluster.
        9. Sorts the terms based on their similarity to the query centroid.
        10. Selects the top 'k' most similar terms and adds them to the expansion terms.
        11. Returns the top 'k' expansion terms.

        Args:
            query (list[str]): The list of query terms to be expanded.
            k (int): The number of expansion terms to be returned.

        Returns:
            list[str]: A list of expanded query terms, including the original query terms and the most similar terms from the collection.
        """

        inv_index = self.collection.inverted_index

        # Get collection indices for each term in query
        indices = [inv_index[term]["id"] for term in query if term in inv_index]

        # if query term not in collection, instantly append in expansion terms
        # Since there is no embedding calculate for it
        expansion_terms = [term for term in query if term not in inv_index]

        # Get query embeddings and drop label (cluster id)
        query_embeddings = self.embeddings.iloc[indices, :-1].values

        if len(indices) > 0:

            # Calculate query point in embedding space by taking the mean
            qv = np.mean(query_embeddings, axis=0)

            # centroids of each cluster
            centroids = (
                self.embeddings.groupby("labels")
                .mean()
                .reset_index()
                .drop(columns=["labels"])
                .values
            )

            # Compute similarity between query and the collection terms in that space
            similarities = cosine_similarity([qv], centroids)[0]

            # Get the cluster assignment for the query
            query_cluster = self.embeddings.iloc[np.argmax(similarities)]["labels"]

            # Find terms from the collection that belong to the same cluster as the query terms
            collection_terms_in_nearest_cluster = [
                inv_index[term]["id"]
                for term, cluster in zip(
                    inv_index.keys(), self.embeddings["labels"].values
                )
                if cluster == query_cluster and term not in query
            ]

            # Calculate the cosine similarity between the query centroid and same cluster terms in the cluster
            query_centroid_similarity = cosine_similarity(
                np.array([qv]),
                self.embeddings.iloc[collection_terms_in_nearest_cluster, :-1].values,
            )

            # Sort the terms based on their similarity to the query centroid
            sorted_indices = np.argsort(-query_centroid_similarity[0])

            # Add the expansion terms from the nearest cluster(s) to the list, selecting the top 'k' terms
            col_terms = list(inv_index.keys())
            for i in sorted_indices[:k]:
                term = col_terms[collection_terms_in_nearest_cluster[i]]
                expansion_terms.append(term)

            # Pick 'k' terms from the expansion terms (if 'k' is greater than the number of expansion terms, take all)
            return expansion_terms[:k]

    def fit_evaluate(self, queries, relevants):

        # inverted index of collection documents
        inv_index = self.collection.inverted_index

        # for each query
        for i, (query, rel) in enumerate(zip(queries, relevants), start=1):

            ################# QUERY EXPANSION ###################
            # k = int(len(query)/2)+1 if len(query) > 12 else len(query)
            k = len(query)  # //2 + 1

            query += self.expand_q(query, k)
            # query += self.expand_q_centroids(query, k)

            # apply apriori to find frequent termsets
            freq_termsets = apriori(query, inv_index, min_freq=1)

            print(
                f"Query {i}/{len(queries)}: len = {len(query)}, frequent = {len(freq_termsets)}"
            )

            # vectorized query generated by apriori
            idf_vec = self.query2vec(freq_termsets)  # (1 X len(termsets)) vector

            # vectorized documents generated by apriori query
            tsf_ij = self.termsets2vec(freq_termsets)  # (len(termsets) X N) matrix

            # model adjusting weight
            weights = self._model_func(freq_termsets)

            # document - termset matrix - model balance weight
            dtsm = self._vectorizer(tsf_ij, idf_vec, weights)

            # cosine similarity between query and every document
            qd_sims = self.qd_similarities(idf_vec, dtsm)

            # rank them in desc order
            retrieved_docs = self.rank_documents(qd_sims)

            # precision | recall of ranking
            pre, rec = precision_recall(retrieved_docs.keys(), rel)

            print(f"=> Query {i}/{100}, precision = {pre:.3f}, recall = {rec:.3f}")

            self.precision.append(round(pre, 3))
            self.recall.append(round(rec, 3))

        return np.array(self.precision), np.array(self.recall)

    def _cnwk(self):
        # Dictionary to store computed _cnwk values for each cluster
        cluster_cnwk = {}

        for term in self.collection.inverted_index:
            # Get the cluster of the current term
            cluster = self.graph.nodes[term]["cluster"]

            # Check if _cnwk value for the cluster has been computed before
            if cluster not in cluster_cnwk:
                _cnwk = 0
                cluster_size = 0

                # Iterate over nodes and compute _cnwk for the cluster
                for node, attrs in self.graph.nodes(data=True):
                    if attrs["cluster"] == cluster:
                        cluster_size += 1
                        _cnwk += self.collection.inverted_index[node]["nwk"]

                # Compute average _cnwk value for the cluster
                cluster_cnwk[cluster] = round(_cnwk / cluster_size, 3)

            # Assign the computed _cnwk value to the current term
            self.collection.inverted_index[term]["cnwk"] = cluster_cnwk[cluster]

        return

    def _model_func(self, termsets):

        inv_index = self.collection.inverted_index
        tns = zeros(len(termsets), dtype=float)

        for i, termset in enumerate(termsets):
            tw = 1
            for term in termset:
                if term in inv_index:
                    # get the cnw of term k and multiply it to total
                    tw *= inv_index[term]["cnwk"]
            tns[i] = tw

        return tns
