### Imports ###

# Imports from standard Python libraries
import json

# Imports from third party libraries
import pandas as pd
import torch
import yaml
from google.cloud import bigquery
from metabeaver.GoogleCloudPlatform.BigQuery.getTableData import get_first_n_rows
from sklearn.neighbors import BallTree
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

### End of Imports ###

### Function Definition ###

# Takes a list containing sentence(s) and encodes them to tokens, them creates vectors representing sentence features.
def embedSentences(sentences: list,
                   tokenizer='sentence-transformers/all-MiniLM-L6-v2', # Name of tokenizer OR path to local file.
                   model='sentence-transformers/all-MiniLM-L6-v2', # Name of model OR path to local file.
                   use_local=False, # Whether to try finding the model on HuggingFace or load from local location.
                   padding=True,
                   truncation=True, # Whether to truncate token sequence above input length of 256.
                   max_length=256 #Default max length of token sequence to consider.
                   ):

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):

        # Get token embeddings from the output of a model. First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # Load AutoModel from huggingface model repository
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=use_local)
    model = AutoModel.from_pretrained(model, local_files_only=use_local)

    # Tokenize sentences.
    encoded_input = tokenizer(sentences,
                              padding=padding,
                              truncation=truncation,
                              max_length=max_length,
                              return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean(average) pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Having converted sentences to a numerical representation of their semantics, return the vectors for each sentence.
    return sentence_embeddings

# Get X rows of data, according to the GCP details exposed in the config-yaml file.
def getData(x):

    # Get the credentials for the API calls
    with open('config-yaml.yaml', 'r') as file:
        configuration = yaml.safe_load(file)

    # Get the path of the credentials and the project name for dynamically requesting a GCP BigQuery client
    credentials_path = configuration['GCP']['credentials_file_loc']
    gcp_project = configuration['GCP']['project_name']

    # Load the credentials from the file path location specified in the yaml.
    with open(credentials_path, 'r') as file:
        credentials_info = json.load(file)

    # Use the 'from_service_account_info' method to dynamically load credentials from a file
    client = bigquery.Client.from_service_account_info(
        credentials_info,
        project=gcp_project
    )

    # Get the first x rows from the table.
    df = get_first_n_rows(client,
                          configuration['GCP']['project_name'],
                          configuration['GCP']['dataset'],
                          configuration['GCP']['table_name'],
                          x)

    return df


# This function generates a summary from an article, using the t5-small pre-trained transformer
def generate_summary(article, # Text to summarise
                     tokenizer_name='t5-small', # Tokenizer to convert text to integer sequence
                     model_name='t5-small', # Model to response to token sequence with and generate response text for this sequence
                     max_length=512, # Max length of input to consider
                     num_beams=4, #
                     length_penalty=1.5, # Penalty on verbosity of response
                     repetition_penalty=1.25, # Penalty to repeating tokens in the response
                     early_stopping=True # Whether we hit the EndOfStream token more readily to prevent verbosity
                     ):

    # Load a T5Tokenizer. Defaults to t5-small unless overridden.
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    # Load a T5Model. Default to t5-small unless overridden.
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Tokenize the string.
    inputs = tokenizer("summarize: " + article, return_tensors="pt", max_length=512, truncation=True)

    # Generate a response to the string. The response should be a summary of the cell content, given the 'summarise: '
    summary_ids = model.generate(inputs["input_ids"],
                                 max_length=max_length,
                                 num_beams=num_beams,
                                 length_penalty=length_penalty,
                                 repetition_penalty=repetition_penalty,
                                 early_stopping=early_stopping)

    # Decode the sequence of with the tokenizer.decode to actual human readable text.
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Return summary of the article from the model.
    return summary

### End of Function Definition ###

### Frozen Variables ###

# Get the data from the cloud and add a column of summarisations using https://huggingface.co/google-t5/t5-small
rows = 1000

### End of Frozen Variables ###

### Dynamic Logic ###

df = getData(rows)
df['Summary'] = df['Page Text'].apply(lambda x: generate_summary(x, 't5-small', 't5-small'))

# Take the summaries, and create a column, which is a 384 dimension vector, representing semantics of each summary.
sentences = df['Summary'].to_list()
encodedSentences = embedSentences(sentences, use_local=False) # Defaults to sentence-transformers/all-MiniLM-L6-v2

# Create a dataframe with columns for tensor values and then join the tensors with the Page, Date, Time web data index.
columns = [f'Tensor_Value_{i}' for i in range(encodedSentences.shape[1])]
tensorFrame = pd.DataFrame(encodedSentences.numpy(), columns=columns)
pagesWithEmbeddings = pd.concat([df, tensorFrame.reset_index(drop=True)], axis=1)

# Convert the PyTorch tensor to a NumPy array
vectors_np = encodedSentences.numpy()

# Build a Ball Tree for the vectors
ball_tree = BallTree(vectors_np)

# Example query vector
#query_vector = torch.randn((384,), dtype=torch.float32).numpy()
query_vector = vectors_np[0]

# Perform nearest neighbor search for the query vector
distances, indices = ball_tree.query([query_vector], k=10)

# Display the results
print("Query Vector:", query_vector)
print("Nearest Neighbor Distances:", distances)
print("Nearest Neighbor Indices:", indices)

### End of Dynamic Logic ###
























