import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import get_label_mappings, get_representation, get_sense_inventory




def build_centroids(embeddings, training_data, id_to_gold):
    labels_to_vecs, labels_to_ids, ids_to_labels, labels_to_freq = get_label_mappings(embeddings, training_data,id_to_gold)

    print("getting centroids")
    centroids = {}
    for label_id, vec in tqdm(labels_to_vecs.items()):
        freq = labels_to_freq[label_id]
        centroids[ids_to_labels[label_id]] = vec / freq

    return centroids



def calculate_closest_sense(word, target_sentence, model, tokenizer,sense_database, sense_centroids, layer, inventory):

    target_word = word['word']
    target_index = target_sentence.index(target_word)
    target_lemma = word['lemma'].lower()
    representation = get_representation(target_sentence, target_index, model, tokenizer)[layer]
  
    sense_inventory_a, sense_inventory_n, sense_inventory_v, sense_inventory_r = inventory

    centroids = []
    labels = []
    pos = word['pos']
    if pos.lower()[0] == "v" and target_lemma in sense_inventory_v.keys():
         candidates = sense_inventory_v[target_lemma]
    elif pos.lower() == "adv" and target_lemma in sense_inventory_r.keys():
        candidates = sense_inventory_r[target_lemma]
    elif pos.lower()[0] == "n" and target_lemma in sense_inventory_n.keys():
        candidates = sense_inventory_n[target_lemma]
    elif pos.lower() == "adj" and target_lemma in sense_inventory_a.keys():
        candidates = sense_inventory_a[target_lemma]
    else:
        candidates = []

    for candidate in candidates:
        if candidate in sense_database:
            centroids.append(sense_centroids[sense_database.index(candidate)])
            labels.append(candidate)

    if len(candidates) == 0:
        similarities = cosine_similarity([representation], sense_centroids)
        closest_cluster = np.argmax(similarities)

        return sense_database[closest_cluster]

    if  len(centroids) != 0:
        similarities = cosine_similarity([representation], centroids)
        closest_cluster = np.argmax(similarities)

        return labels[closest_cluster]

    return candidates[0]

def evaluate_to_file(embeddings, test_data, model, tokenizer, layer, training_data, id_to_gold, path='output.key'):
    responds = {}
    centroids = build_centroids(embeddings, training_data, id_to_gold)
    sense_database = list(centroids.keys())
    sense_centroids = np.stack([centroids[s] for s in sense_database])
    inventory = get_sense_inventory()

    for element in tqdm(test_data):
      for word in element['words']:
        result = calculate_closest_sense(word, element['sentence_text'], model, tokenizer, sense_database, sense_centroids, layer, inventory)
        responds[word['id']] = [result]

    with open(path, 'w') as f:
      for instance_id, sense_keys in responds.items():
        f.write(f"{instance_id} {' '.join(sense_keys)}\n")




      


