from numpy import dot, mean
from numpy.linalg import norm


def tf(terms):
    tf = {}
    for term in terms:
        if term not in tf:
            tf[term] = 1
        elif term in tf:
            tf[term] += 1
    return tf


def cosine_similarity(u, v):
    if (u == 0).all() | (v == 0).all():
        return 0.
    else:
        return dot(u,v) / (norm(u)*norm(v))


def precision_recall(retrieved_docs, relevant):
    
    total, retrieved = 0, 1
    precision, recall = [], []

    # for each document model retrieved
    for doc in retrieved_docs:

        # if it's considered as relevant
        if doc in relevant:
            # count it
            total += 1
            # calculate current precision | recall
            p = total / retrieved
            r = total / len(relevant)

            precision += [p]
            recall += [r]

        # count total retrieved
        retrieved += 1

    # return mean precision and recall for given query
    return mean(precision), mean(recall)