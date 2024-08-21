import json

# Load the relations dictionary from the JSON file
input_file_path = "conceptnet.json"
with open(input_file_path, "r") as json_file:
    relations_dict = json.load(json_file)

# Define the specific language to look for
specific_language = "mt"

# Check if the specific language exists in the dictionary
if specific_language in relations_dict:
    triples = relations_dict[specific_language]
    print(triples)
    print(f"Language: {specific_language} - Number of triples: {len(triples)}\n")

else:
    print(f"Language {specific_language} not found in the dataset.")
