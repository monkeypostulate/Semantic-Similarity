from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, AlbertTokenizer, ElectraTokenizer
from sentence_transformers import SentenceTransformer


import numpy as np


def word_vector(text, tokenizer = 'bert'):
    if tokenizer == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif tokenizer == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif tokenizer == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif tokenizer == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        
    tokens = tokenizer.tokenize(text)
    text_embedding = tokenizer.convert_tokens_to_ids(tokens)

    return tokens, text_embedding


# ########################################################
#

def documents_to_token_ids(data, column, tokenizer = 'bert', max_vector = 500):
    
    number_documents = data.shape[0]
    document_embedding = np.zeros((number_documents, max_vector))
    i = 0

    for text in data[column]:
        temp_tokens, temp_embedding = word_vector(text, tokenizer = tokenizer)

        vect_len = np.min([len(temp_embedding), 500])
        document_embedding[i,0:vect_len] = temp_embedding[0:vect_len]
        i += 1
    
    return document_embedding
    
    
    

def documents_to_vector(documents ,
                        model = 'all-mpnet-base-v2',
                        filename = None,
                        max_seq_length = 500):
    """
    -----------------------------
    Parameters
    documents: list of strings to be vectorise

    model:
    -----------------------------
    Output:
    numpy array. 
    """
    # vectorise documents
    embedding_model = SentenceTransformer(model)
    embedding_model.max_seq_length = max_seq_length
    
    temp_embedding = embedding_model.encode(sentences = documents,
                             show_progress_bar = True)
    
    return temp_embedding
 