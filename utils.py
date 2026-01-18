import torch
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from nltk.corpus import wordnet as wn

import os
import pickle

def get_data(path, id_to_gold=None, train=True):
  tree = ET.parse(path)
  root = tree.getroot()
  training_data = []
  sense_data = {}
  for i, text in enumerate(root.findall('text')):
    for sentence in text.findall('sentence'):
      sentence_text = []
      words = []
      for elem in sentence:
        if elem.tag == 'wf':
          sentence_text.append(elem.text)
        elif elem.tag == 'instance':
          sentence_text.append(elem.text)
          words.append({
            'word': elem.text,
            'lemma': elem.get('lemma'),
            'pos': elem.get('pos'),
            'id': elem.get('id')
            })
      training_data.append({
        'sentence_text': sentence_text,
        'words': words
      })
      if train:
        for word in words:
          senses = id_to_gold[word["id"]]
          for sense in senses:
            if sense in sense_data:
              sense_data[sense].append({"sentence": sentence_text, "word": word["word"]})
            else:
              sense_data[sense] = [{"sentence": sentence_text, "word": word["word"]}]
  return training_data, sense_data

def get_labels(key_file):
        id_to_gold, sense_to_id = {}, {}
        with open(key_file) as f:
            for l in f:
                position_id, *senses = l.split()
                id_to_gold[position_id] = senses

                for s in senses:
                    if s not in sense_to_id:
                        sense_to_id[s] = [len(sense_to_id), 1]
                    else:
                        sense_to_id[s][1] += 1
        return id_to_gold, sense_to_id


def load_or_compute_embeddings(
    model_name,
    layers,
    compute_fn,
    compute_args,
    prefix=""
):
    paths = [
        f"/data/somas/repr/{model_name}{prefix}_{layer}.npy"
        for layer in layers
    ]

    if all(os.path.exists(p) for p in paths):
        emb = np.array([np.load(p) for p in paths])
    else:
        emb = compute_fn(*compute_args)
        for i, layer in enumerate(layers):
            np.save(paths[i], emb[i])

    return emb

def get_sense_inventory():
    
    sense_inventory_a = {}
    sense_inventory_n = {}
    sense_inventory_v = {}
    sense_inventory_r = {}

    # Map your POS labels to WordNet POS labels
    pos_map = {
        "n": wn.NOUN,
        "v": wn.VERB,
        "a": wn.ADJ,
        "r": wn.ADV
    }

    # For each POS
    for pos, wn_pos in pos_map.items():

        # Iterate all lemmas that exist in WordNet for that POS
        for lemma in wn.all_lemma_names(pos=wn_pos):
            lemma_l = lemma.lower()

            # Get all WordNet sensekeys for lemma+pos
            sensekeys = [l.key() for l in wn.lemmas(lemma, wn_pos)]

            # Store in correct POS dictionary
            if pos == "a":
                sense_inventory_a[lemma_l] = sensekeys
            elif pos == "n":
                sense_inventory_n[lemma_l] = sensekeys
            elif pos == "v":
                sense_inventory_v[lemma_l] = sensekeys
            elif pos == "r":
                sense_inventory_r[lemma_l] = sensekeys

    return (
        sense_inventory_a,
        sense_inventory_n,
        sense_inventory_v,
        sense_inventory_r
    )

def get_wn_reprs():
    res = []
    res_s = []
    all_sense_keys = []
    i = 0
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            try:
                key = lemma.key()
                all_sense_keys.append(key)
            except:
                continue

    for sense_key in tqdm(all_sense_keys):

        try:
            synset = wn.lemma_from_key(sense_key).synset()
        except:
            continue

        lemmas = [l.name().replace("_", " ").lower() for l in synset.lemmas()]
        gloss = synset.definition()
        examples = synset.examples()

        wn_reps = []


        for ex in examples:
            tokens = ex.split()
            match_pos = []

            # find lemma occurrences
            for lemma in lemmas:
                for i, tok in enumerate(tokens):
                    if lemma in tok.lower():
                        match_pos.append(i)

            for pos in match_pos:
                rep = get_representation(tokens, pos, model, tokenizer)
                wn_reps.append(rep)

        if gloss: #and len(wn_reps) == 0:
            gloss_tokens = gloss.split()
            target_idx = min(len(gloss_tokens) - 1, 0)
            rep = get_representation(gloss_tokens, target_idx, model, tokenizer)
            wn_reps.append(rep)
        # Add representations to label vectors
        for rep in wn_reps:
            res.append(rep[min(layers):])
            res_s.append(sense_key)
            i+=1


    res = np.array(res)

    res_embeddings = np.stack([res[:, i, :] for i in layers])

    return res_embeddings, res_s


def load_or_compute_wn(model_name, layers):
    emb_paths = [
        f"/data/somas/repr/{model_name}_wn_{layer}.npy"
        for layer in layers
    ]
    senses_path = f"/data/somas/repr/{model_name}_wn_senses.pkl"

    if all(os.path.exists(p) for p in emb_paths) and os.path.exists(senses_path):
        wn_emb = np.array([np.load(p) for p in emb_paths])
        with open(senses_path, "rb") as f:
            wn_senses = pickle.load(f)
    else:
        wn_emb, wn_senses = get_wn_reprs()
        for i, layer in enumerate(layers):
            np.save(emb_paths[i], wn_emb[layer])
        with open(senses_path, "wb") as f:
            pickle.dump(wn_senses, f)

    return wn_emb, wn_senses

def get_label_mappings(M, training_data, id_to_gold):
    labels_to_vecs = {}
    labels_to_ids, ids_to_labels = {}, {}
    labels_to_freq = []

    idx = 0 

    for sentence in training_data[0]:  
        for token in sentence['words']:
            vec = M[idx]  
            if vec.sum() > 0:
                vec /= vec.sum()
            idx += 1
            senses = id_to_gold[token["id"]]

            for label in senses:
                if label not in labels_to_ids:
                    label_id = len(labels_to_ids)
                    ids_to_labels[label_id] = label
                    labels_to_ids[label] = label_id
                    labels_to_freq.append(1)
                    labels_to_vecs[label_id] = vec
                else:
                    labels_to_freq[labels_to_ids[label]] += 1
                    labels_to_vecs[labels_to_ids[label]] += vec
    
    if training_data[1]:
        for sentence in training_data[1]:  
            for token in sentence['words']:
                vec = M[idx]  
                if vec.sum() > 0:
                    vec /= vec.sum()
                idx += 1
                senses = id_to_gold[token["id"]]

                for label in senses:
                    if label not in labels_to_ids:
                        label_id = len(labels_to_ids)
                        ids_to_labels[label_id] = label
                        labels_to_ids[label] = label_id
                        labels_to_freq.append(1)
                        labels_to_vecs[label_id] = vec
                    else:
                        labels_to_freq[labels_to_ids[label]] += 1
                        labels_to_vecs[labels_to_ids[label]] += vec

    
    if training_data[2]:
        for sense in training_data[2]:  
            vec = M[idx]  
            if vec.sum() > 0:
                vec /= vec.sum()
            idx += 1

            if sense not in labels_to_ids:
                label_id = len(labels_to_ids)
                ids_to_labels[label_id] = sense
                labels_to_ids[sense] = label_id
                labels_to_freq.append(1)
                labels_to_vecs[label_id] = vec
            else:
                labels_to_freq[labels_to_ids[sense]] += 1
                labels_to_vecs[labels_to_ids[sense]] += vec

    assert idx == M.shape[0], f"Used {idx} rows, but M has {M.shape}"

    return labels_to_vecs, labels_to_ids, ids_to_labels, labels_to_freq

def get_train_embeddings(model, tokenizer, training_data_p):
    res = []

    print("processing train embeddings")
    for sentence in tqdm(training_data_p):
        for word in sentence["words"]:
            embedding = get_representation(sentence["sentence_text"], sentence["sentence_text"].index(word["word"]), model, tokenizer)
            res.append(embedding[min(layers):])

    res = np.array(res)
    res_embeddings = np.stack([res[:, i, :] for i in layers])
    return res_embeddings

def tokenize_sequence(sequence, tokenizer):
        orig_to_tok_map, transformer_tokens = [], []
        for tok_pos, orig_token in enumerate(sequence):
            orig_to_tok_map.append(len(transformer_tokens))
            transformer_tokens.extend(tokenizer.tokenize('{}{}'.format(' ' if tok_pos>0 else '', orig_token)))


        orig_to_tok_map.append(len(transformer_tokens))

        indexed_tokens = tokenizer.convert_tokens_to_ids(transformer_tokens)
        indexed_tokens_with_specials = tokenizer.build_inputs_with_special_tokens(indexed_tokens)

        if len(indexed_tokens) == 0:
            return None, None

        specials_added = indexed_tokens_with_specials.index(indexed_tokens[0])
        orig_to_tok_map = [x + specials_added for x in orig_to_tok_map]
        return orig_to_tok_map, indexed_tokens_with_specials

def get_representation(sentence, index, model, tokenizer, pooling_strategy = "mean"):
  
    orig_to_tok_map, indexed_tokens_with_specials = tokenize_sequence(sentence, tokenizer)

    tokenized_start_index = orig_to_tok_map[index]
    tokenized_end_index = orig_to_tok_map[index+1]
    with torch.no_grad():
        outputs = model(torch.tensor([indexed_tokens_with_specials]))
        hidden_states = torch.stack(outputs.hidden_states)
    hidden_states = hidden_states[:, 0]

    if pooling_strategy == "mean":
        return hidden_states[:, tokenized_start_index:tokenized_end_index].mean(dim=1).cpu().numpy()  # (num_layers, hidden_dim)
    elif pooling_strategy == "first":
        return hidden_states[:, tokenized_start_index].cpu().numpy()  # (num_layers, hidden_dim)
    elif pooling_strategy == "last":
        return hidden_states[:, tokenized_end_index - 1].cpu().numpy()
    
def preprocess_data(embeddings):
    np_embeddings = np.array(embeddings)
    np_embeddings = row_normalize(np_embeddings)
    np_embeddings = np_embeddings.T
    if not np.isfortran(np_embeddings):
        np_embeddings = np.asfortranarray(np_embeddings)
    return np_embeddings

def row_normalize(embeddings):
    row_norms = np.sqrt((embeddings**2).sum(axis=1))[:, np.newaxis]
    row_norms[row_norms==0] = 1  # we do not want to divide by 0
    return embeddings / row_norms