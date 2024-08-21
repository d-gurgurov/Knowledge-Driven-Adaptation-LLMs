import json

# Load the relations dictionary from the JSON file
input_file_path = "conceptnet.json"
with open(input_file_path, "r") as json_file:
    relations_dict = json.load(json_file)

# Prepare the results dictionary
results = {}
num_languages = len(relations_dict)
results["number_of_languages"] = num_languages

for language, triples in relations_dict.items():
    # Initialize language data
    language_data = {
        "num_triples": len(triples),
        "unique_relationships": {},
        "connected_languages": {}
    }
    
    for triple in triples:
        start_node_uri, relation_uri, end_node_uri = triple

        # Count the relationships
        if relation_uri not in language_data["unique_relationships"]:
            language_data["unique_relationships"][relation_uri] = 0
        language_data["unique_relationships"][relation_uri] += 1

        # Extract and track connected languages and their counts
        target_language = end_node_uri.split("/")[2]
        if target_language not in language_data["connected_languages"]:
            language_data["connected_languages"][target_language] = 0
        language_data["connected_languages"][target_language] += 1

    # Add the language data to the results
    results[language] = language_data

# Write the results to a JSON file
output_file_path = "conceptnet_stats.json"
with open(output_file_path, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"Summary saved to {output_file_path}")
