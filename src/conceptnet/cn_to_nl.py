from sklearn.model_selection import train_test_split
import json
import pandas as pd

# Load relationship mapping from ConceptNet
relationship_mapping = {
    'Antonym': 'is the opposite of',
    'DerivedFrom': 'is derived from',
    'EtymologicallyDerivedFrom': 'is etymologically derived from',
    'EtymologicallyRelatedTo': 'is etymologically related to',
    'FormOf': 'is a form of',
    'PartOf': 'is a part of',
    'HasA': 'belongs to',
    'UsedFor': 'is used for',
    'AtLocation': 'is a typical location for',
    'Causes': 'causes',
    'CausesDesire': 'makes someone want',
    'MadeOf': 'is made of',
    'ReceivesAction': 'receives action of',
    'HasSubevent': 'is a subevent of',
    'HasFirstSubevent': 'is an event that begins with subevent',
    'HasLastSubevent': 'is an event that concludes with subevent',
    'HasPrerequisite': 'has prerequisite of',
    'HasProperty': 'can be described as',
    'MotivatedByGoal': 'is a step toward accomplishing the goal',
    'ObstructedBy': 'is an obstacle in the way of',
    'Desires': 'is a conscious entity that typically wants',
    'CreatedBy': 'is a process or agent that creates',
    'CapableOf': 'is capable of',
    'HasContext': 'is a word used in the context of',
    'IsA': 'is a type of',
    'RelatedTo': 'is related to',
    'SimilarTo': 'is similar to',
    'Synonym': 'is a synonym of',
    'SymbolOf': 'symbolically represents',
    'DefinedAs': 'is a more explanatory version of',
    'DistinctFrom': 'is distinct from',
    'MannerOf': 'is a specific way to do',
    'LocatedNear': 'is typically found near',
}

# Function to load and process ConceptNet data
def load_conceptnet_data(data):
    entities = []
    for language, triples in data.items():
        for triple in triples:
            start = triple[0].split('/')[-1].replace("_", " ")
            rel = triple[1].split('/')[-1]
            end = triple[2].split('/')[-1].replace("_", " ")

            if start != end:
                entities.append((start, rel, end))
    return entities

# Function to construct natural language sentences from triples
def construct_sentences(triples):
    sentences = []
    for triple in triples:
        subject, relationship, obj = triple
        if relationship in relationship_mapping:
            sentence = f"{subject} {relationship_mapping[relationship]} {obj}."
            sentences.append(sentence)
    return sentences

# List of languages
languages_to_process = ['am', 'uz', 'su', 'cy', 'mr', 'te', 'ku', 'mk', 'bn', 'ka', 'sk', 'el', 'th', 'az', 'lv', 'sl', 'he', 'ro', 'da', 'ur', 'si', 'yo', 'sw', 
             'ug', 'bo', 'mt', 'jv', 'ne', 'ms', 'bg']

# Load the data
with open(f"/netscratch/dgurgurov/thesis/data/conceptnet/conceptnet.json") as f:
    data = json.load(f)

# Process only the languages in the specified list
for language in languages_to_process:
    if language in data:
        # Extract triples and construct sentences
        cn_triples = load_conceptnet_data({language: data[language]})
        cn_sents = construct_sentences(cn_triples)

        # Split the sentences into 80% training, 10% validation, 10% test sets
        train_sents, temp_sents = train_test_split(cn_sents, test_size=0.2, random_state=42)
        val_sents, test_sents = train_test_split(temp_sents, test_size=0.5, random_state=42)

        # Create DataFrames
        train_df = pd.DataFrame(train_sents, columns=["text"])
        val_df = pd.DataFrame(val_sents, columns=["text"])
        test_df = pd.DataFrame(test_sents, columns=["text"])

        # Save the data into CSV files
        train_df.to_csv(f"/netscratch/dgurgurov/thesis/data/conceptnet/train_cn_{language}.csv", index=False)
        val_df.to_csv(f"/netscratch/dgurgurov/thesis/data/conceptnet/val_cn_{language}.csv", index=False)
        test_df.to_csv(f"/netscratch/dgurgurov/thesis/data/conceptnet/test_cn_{language}.csv", index=False)

        print(f"Processed language: {language}, Number of triples: {len(cn_triples)}")
    else:
        print(f"Language {language} not found in the dataset.")
