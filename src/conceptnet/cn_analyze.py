import json

# Load the relations dictionary from the JSON file
input_file_path = "conceptnet.json"
with open(input_file_path, "r") as json_file:
    relations_dict = json.load(json_file)

# Prepare the results
results = []
num_languages = len(relations_dict)
results.append(f"Number of languages: {num_languages}\n")

for language, triples in relations_dict.items():
    # Number of triples
    num_triples = len(triples)
    results.append(f"Language: {language} - Number of triples: {num_triples}\n")

    # Unique relationships and their counts
    relationship_counts = {}
    connected_languages = {}
    
    for triple in triples:
        start_node_uri, relation_uri, end_node_uri = triple

        # Count the relationships
        if relation_uri not in relationship_counts:
            relationship_counts[relation_uri] = 0
        relationship_counts[relation_uri] += 1

        # Extract and track connected languages and their counts
        target_language = end_node_uri.split("/")[2]
        if target_language not in connected_languages:
            connected_languages[target_language] = 0
        connected_languages[target_language] += 1

    # Add unique relationships and their counts to the results
    results.append("  Unique relationships:\n")
    for relation, count in relationship_counts.items():
        results.append(f"    {relation}: {count} triples\n")

    # Add connected languages and the count of triples for each to the results
    num_connected_languages = len(connected_languages)
    results.append(f"  Connected to {num_connected_languages} different languages:\n")
    for target_lang, count in connected_languages.items():
        results.append(f"    {target_lang}: {count} triples\n")

    results.append("\n")  # Add a blank line between languages

# Write the results to a text file
output_file_path = "conceptnet_stats.txt"
with open(output_file_path, "w") as txt_file:
    txt_file.writelines(results)

print(f"Summary saved to {output_file_path}")
