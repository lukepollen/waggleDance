### Imports ###

from transformers import AutoTokenizer, AutoModelForCausalLM

### End of Imports ###

# Download from https://huggingface.co/google/gemma-2b/tree/main . Afterm change this to your download location.
twoBConfigPath = 'C:/Users/lukep/OneDrive/workandplay/Logic and Software/Computer Science/LLM/gemma/2b'

# Load the main components to respond to our text, a tokenizer to convert text to numeric, and a model to reverse.
tokenizer = AutoTokenizer.from_pretrained(twoBConfigPath, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(twoBConfigPath, local_files_only=True)

# Test query to test Gemma with. MMMM, delicious olives and pineapple toppings, force strong in it is.
input_text = "What is the derivative of X to the power of 2?"

# Convert the text to a tensor of integers and then send to the model to get a response.
input_ids = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(**input_ids, max_length=512) # Tweak max_length for longer response.
print(tokenizer.decode(outputs[0])) # Print the response.