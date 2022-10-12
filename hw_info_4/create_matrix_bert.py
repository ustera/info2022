from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
import torch
from tqdm import tqdm
import numpy as np


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def indexing_bert(text):
    """ Imports tokenizer, model and counts BERT matrix  """
    global tokenizer
    global model
    batches = []
    for batch in np.array_split(list(text), 100):
        batches.append(list(batch))
    tensors = tuple()
    for batch in tqdm(batches):
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        batch_matrix = mean_pooling(model_output, encoded_input['attention_mask'])
        tensors += (batch_matrix,)
    bert_matrix = normalize(torch.vstack(tensors))
    return bert_matrix, tokenizer, model


tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")


