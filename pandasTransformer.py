### Imports ###

# Imports from standard Python libraries
import json

# Imports from third party libraries
import yaml
from google.cloud import bigquery
from metabeaver.GoogleCloudPlatform.BigQuery.getTableData import get_first_n_rows
from transformers import T5Tokenizer, T5ForConditionalGeneration

### End of Imports ###


### Function Definition ###

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
                     tokenizer, # Tokenizer to convert text to integer sequence
                     model, # Model to response to token sequence with and generate response text for this sequence
                     max_length=512, # Max length of input to consider
                     num_beams=4, #
                     length_penalty=1.5, # Penalty on verbosity of response
                     repetition_penalty=1.25, # Penalty to repeating tokens in the response
                     early_stopping=True # Whether we hit the EndOfStream token more readily to prevent verbosity
                     ):

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

# Create a tokenizer and model relating to the t5-small transformer
model_name = "t5-small"

# Number of rows of target table to retrieve from the cloud
x = 100

### End of Frozen Variables ###

### Dynamic Logic ###

# Load the light-weight t5 transformer model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Replace with your dataframe generating function.
df = getData(x)

# Create a summary for every article in the dataframe, using our lightweight pre-trained transformer.
df['summary'] = df['Page Text'].apply(lambda x: generate_summary(x, tokenizer, model))

### End of Dynamic Logic ###


























