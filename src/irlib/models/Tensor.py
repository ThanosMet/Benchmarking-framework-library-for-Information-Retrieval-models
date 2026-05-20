from models.Model import Model
from transformers import BertTokenizer, BertModel
import torch
import os
from pickle import load
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from utilities.document_utls import calc_precision_recall
from tqdm import tqdm
from time import time
from statistics import mean, median

class TensorModel(Model):
    def __init__(self, collection, tensor_path):
        super().__init__(collection)
        # Path to the saved document tensors on disk
        self.tensor_path = tensor_path
    
    def fit(self, queries=None):
            if queries is None:
                queries = self._queries
            print('abnaroz')
            query_sim = [] #RQD for each query
            rel = self._relevant
            # Loop 1 (Single query from Collection)
            start = time()
            for i, query in enumerate(queries):
                # Remove stopwords
                trimmed_query = []
                for word in query:
                        if word.lower() not in stopwords.words('english'):
                            trimmed_query.append(word)
                query = trimmed_query
                # Tokenize Query with Bert and get Embeddings
                print(f'Tokenizing Query[{i}]]')
                query_dict = self._query_tokenize(query)
                print(list(query_dict.keys()))

                # Dict to store relevance of query with every document
                rqd = {} # Relevance Query Document {Document ID: }

                # Loop 2 (Single Document from Collection)
                for doc in tqdm(self.collection.docs):

                    # Load document tensors from disk
                    with open(os.path.join(self.tensor_path, str(doc.doc_id)), 'rb') as tensorfile:
                        document_dict = load(tensorfile)
                    
                    # List to store relevance
                    rqtd = [] # Relevance Query token Document
                    
                    # Loop 3 (Single token from query)
                    for q_key, q_value in query_dict.items():
                        # List to store relevance of query token with document token [sim(d_key, q_key)]
                        rqtdt = [] # sim(DTok, QTok)

                        # Loop 4 (Single token from document)
                        for d_key, d_value in document_dict.items():
                            qtdt_similarity = cosine_similarity(torch.reshape(q_value, (1, -1)), torch.reshape(d_value, (1, -1)))
                            rqtdt.append(qtdt_similarity[0][0]) # sim(Dtok, QTok) for all Dtok of D
                        rqtd.append(max(rqtdt)) # Document ID : Relevance with Query token
                    rqd[doc.doc_id] = mean(rqtd)
                query_sim.append(rqd)

                # break

            end = time()
            precision = []
            for doc_sim, relevant_docs in zip(query_sim, rel):
                    document_similarities = {id: sim for id, sim in sorted(doc_sim.items(), key=lambda item: item[1], reverse=True)}
                    k = len(document_similarities.keys())
                    pre, rec, mrr = calc_precision_recall(document_similarities.keys(), relevant_docs, k)
                    print(round(pre, 8))
                    precision.append(round(pre, 8))
            print(f'Total time: {((end - start) / 60):.2f} minutes')
            print(mean(precision))
            return None

    def get_model(self):
        pass
    
    def _model_func(self):
        pass

    def _vectorizer(self):
        pass

    # Copipasta of TokDocument.doc_tokenize()
    def _query_tokenize(self, query):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        encoding = tokenizer.__call__(
            query,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            is_split_into_words=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        # Tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state
        # Embeddings
        tensors = word_embeddings[0]
        tensor_list = []
        for i in range(len(tensors)):
            tensor_list.append(tensors[i])
        # Aggregate the tensors of every unique token and save as dictionary {token: tensor}
        aggregate_tensors = {}
        for token, tensor in zip(tokens, tensor_list):
            if token not in aggregate_tensors:
                aggregate_tensors[token] = tensor
            elif token in aggregate_tensors:
                current_tensor = aggregate_tensors[token]
                new_tensor = torch.mean(torch.stack((current_tensor, tensor)), dim=0)
                aggregate_tensors[token] = new_tensor
        return aggregate_tensors



















# Works (in theory), mega slow (1+ hour per Query token)

# def fit(self, queries=None):
#         if queries is None:
#             queries = self._queries
#         print('abnaroz')
#         # Loop 1 (Single query from list)
#         for query in queries:
#             # Tokenize Query with Bert and get Embeddings
#             print('Tokenizing Query...')
#             query_dict = self._query_tokenize(query)
#             print(list(query_dict.keys()))

#             # List to store relevance of query with every document
#             rqd = []

#             # Loop 2 (Single token from query)
#             for q_key, q_value in query_dict.items():
#                 print(f'Calculating Relevance for Query token: {q_key}')
#                 token_time_start = time()
#                 # Dict to store relevance of query token with every document {document_id: relevance}
#                 rqtd = {} # Relevance Query token Document

#                 # Loop 3 (Single Document from Corpus)
#                 for doc in tqdm(self.collection.docs):
#                     # List to store relevance of query token with document token [sim(d_key, q_key)]
#                     rqtdt = [] #Relevance Query token Document token

#                     # Load document tensors from disk
#                     with open(os.path.join(self.tensor_path, str(doc.doc_id)), 'rb') as tensorfile:
#                         document_dict = load(tensorfile)

#                     # Loop 4 (Single token from document)
#                     for d_key, d_value in document_dict.items():
#                         qtdt_similarity = cosine_similarity(torch.reshape(q_value, (1, -1)), torch.reshape(d_value, (1, -1)))
#                         rqtdt.append(qtdt_similarity[0][0])
#                     max_rqtdt = max(rqtdt)
#                     rqtd[doc.doc_id] = max_rqtdt
                
#                 rqd.append(rqtd)
#                 token_time_end = time()
#                 print(f'Time for {q_key}: {(token_time_end - token_time_start):.2f} seconds')
#             # Final Relevance Dictionary for entire query
#             relevance_lists = {}
#             for relevance_dict in rqtd:
#                 for r_key, r_value in relevance_dict:
#                     if r_key not in relevance_lists:
#                         relevance_lists[r_key] = []
#                     relevance_lists[r_key].append(r_value)
            
#             relevance_final = {}
#             for f_key, f_value in relevance_lists:
#                 relevance_final[f_key] = sum(f_value) / len(f_value)


                
#             break # only test for first query
#         return None




# def fit(self, queries=None):
#         if queries is None:
#             queries = self._queries
#         print('abnaroz')
#         query_sim = [] #RQD for each query
#         rel = self._relevant[:1]
#         loops = 0
#         # Loop 1 (Single query from Collection)
#         for query in queries:
#             # Remove stopwords
#             trimmed_query = []
#             for word in query:
#                     if word.lower() not in stopwords.words('english'):
#                         trimmed_query.append(word)
#             query = trimmed_query
#             # Tokenize Query with Bert and get Embeddings
#             print('Tokenizing Query...')
#             query_dict = self._query_tokenize(query)
#             print(list(query_dict.keys()))

#             # Dict to store relevance of query with every document
#             rqd = {} # Relevance Query Document {Document ID: }

#             # Loop 2 (Single Document from Collection)
#             for doc in tqdm(self.collection.docs):

#                 # Load document tensors from disk
#                 with open(os.path.join(self.tensor_path, str(doc.doc_id)), 'rb') as tensorfile:
#                     document_dict = load(tensorfile)
                
#                 # List to store relevance
#                 rqtd = [] # Relevance Query token Document
                
#                 # Loop 3 (Single token from query)
#                 for q_key, q_value in query_dict.items():
#                     # List to store relevance of query token with document token [sim(d_key, q_key)]
#                     rqtdt = [] # sim(DTok, QTok)

#                     # Loop 4 (Single token from document)
#                     for d_key, d_value in document_dict.items():
#                         qtdt_similarity = cosine_similarity(torch.reshape(q_value, (1, -1)), torch.reshape(d_value, (1, -1)))
#                         rqtdt.append(qtdt_similarity[0][0]) # sim(Dtok, QTok) for all Dtok of D
#                     rqtd.append(max(rqtdt)) # Document ID : Relevance with Query token
#                 rqd[doc.doc_id] = mean(rqtd)
#             query_sim.append(rqd)
            
#             loops += 1
#             if loops >= 1:
#                 break

#         precision = []
#         for doc_sim, relevant_docs in zip(query_sim, rel):
#                 document_similarities = {id: sim for id, sim in sorted(doc_sim.items(), key=lambda item: item[1], reverse=True)}
#                 k = len(document_similarities.keys())
#                 pre, rec, mrr = calc_precision_recall(document_similarities.keys(), relevant_docs, k)
#                 print(round(pre, 8))
#                 precision.append(round(pre, 8))
#         print(mean(precision))
#         return None