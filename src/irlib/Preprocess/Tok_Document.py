import os
import re
import pickle

import torch
import nltk
from transformers import BertTokenizer, BertModel

from Preprocess.Document import Document
from utilities.document_utls import calculate_tf
class TokDocument(Document):
    r"""
        Based on the existing Document class, this subclass adds functionality
        needed for tokenizing and encoding the document's text using BERT.
        Intended to be used with the GIRTE class for Information Retrieval tasks.
        
        Args:
            path (`str`, defaults to `''`):
                The path of the document file on the disk.
            bert (`str`, {`'base'` or `'large'`}, defaults to `'base'`):
                Whether to use *bert-base-uncased* or *bert-large-uncased*.
            stopwords (`bool`, defaults to `False`):
                Whether or not to filter stopwords out of the document prior
                to processing. Stopwords defined by nltk.corpus.stopwords('english').
    """

    def __init__(self, path='', bert='base', stopwords=False):
        try:
            self.path = path
        except FileNotFoundError:
            raise FileNotFoundError
        try:
            self.doc_id = int(re.findall(r'\d+', self.path)[0])
        except IndexError:
            self.doc_id = 696969

        self._bert = 'large' if bert == 'large' else 'base'
        self._stopwords = stopwords
        self.terms = self.read_document()
        self.text = ' '.join(self.terms)
        # TF Dictionary: {token: number of occurances in document}
        self.token_frequency = {}
        
        # Generate and save tensors on the disk.
        # Tensor Dictionary: {token: torch.tensor}
        swords = 'sw' if stopwords else 'nsw'
        self.tensor_path = f'C:/picklejar/tensors/{self._bert}/{swords}'
        os.makedirs(self.tensor_path, exist_ok=True)
        with open(os.path.join(self.tensor_path, str(self.doc_id)), 'wb') as picklefile:
            pickle.dump(self.doc_encode(), picklefile)
        
    
    def __str__(self):
        return f'ID: {self.doc_id}\nTerms: {self.terms}'
    
    def doc_encode(self):
        # Initialize and run BERT to generate tokens and embeddings
        tokenizer = BertTokenizer.from_pretrained(f'bert-{self._bert}-uncased')
        model = BertModel.from_pretrained(f'bert-{self._bert}-uncased')
        # Filter out stopwords if applicable
        terms = []
        if self._stopwords == True:
            for word in self.terms:
                if word.lower() not in nltk.corpus.stopwords.words('english'):
                    terms.append(word)
        else:
            terms = self.terms
        encoding = tokenizer.__call__(
            terms,
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
        # Token frequency
        self.token_frequency = calculate_tf(tokens)
        # Aggregate the tensors of every unique token and save as dictionary
        aggregate_tensors = {}
        for token, tensor in zip(tokens, tensor_list):
            if token not in aggregate_tensors:
                aggregate_tensors[token] = tensor
            elif token in aggregate_tensors:
                current_tensor = aggregate_tensors[token]
                new_tensor = torch.mean(torch.stack((current_tensor, tensor)), dim=0)
                aggregate_tensors[token] = new_tensor
        return aggregate_tensors
