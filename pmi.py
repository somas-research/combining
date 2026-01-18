import numpy as np
from tqdm import tqdm
import nltk
import scipy
import spams
from scipy.sparse import csr_matrix
from utils import get_label_mappings, get_representation, get_sense_inventory

nltk.download('wordnet')


def get_eval_embeddings(model, tokenizer, layers, test_data):
    res = []
    print("processing eval embeddings")
    for sentence in tqdm(test_data):
        for word in sentence["words"]:
            embedding = get_representation(sentence["sentence_text"], sentence["sentence_text"].index(word["word"]), model, tokenizer)
            res.append(embedding[min(layers):])
    
    res = np.array(res)
    res_embeddings = []

    for i in layers:
        layer_reprs = res[:,i,:]
        res_embeddings.append(layer_reprs)
        
    return res_embeddings



def get_pmis(mtx):
    total, row_sum, col_sum = mtx.sum(), mtx.sum(axis=1), mtx.sum(axis=0)
    data, indices, ind_ptr = [], [], [0]
    for i, r in enumerate(mtx):
        if np.any(r.data==0):
            zero_idx = np.where(r.data==0)[0]
        row_marginal = row_sum[i,0] + 1e-11
        col_marginal = col_sum[0, r.indices] + 1e-11
        pmis = np.ma.log((total * r.data) / (row_marginal * col_marginal)).filled(0)
        pmis /= -np.ma.log(r.data/total).filled(1)
        idxs, pmi_values = [], []
        for idx in range(len(r.indices)):
            if pmis[0,idx] > 0:
                idxs.append(r.indices[idx])
                pmi_values.append(pmis[0,idx])

        indices.extend(idxs)
        data.extend(pmi_values)
        ind_ptr.append(len(data))
    ind_ptr.append(len(data))
    MA = csr_matrix((data, indices, ind_ptr), shape=(mtx.shape[0]+1, mtx.shape[1]))
    return MA



def evaluate_to_file(embedding, eval_embedding, K=3000, iter = 100, lambda1=0.05, batchsize=400, training_data=None, id_to_gold=None, test_data=None, path='output.key'):

    params = {'K': K, 'lambda1': lambda1, 'numThreads': 8, 'iter': iter, 'batchsize': batchsize, 'posAlpha': True, 'verbose': False}

    D = spams.trainDL(embedding, **params)

    lasso_params = {x:params[x] for x in ['L','lambda1','lambda2','mode','pos','ols','numThreads','length_path','verbose'] if x in params}
    lasso_params['pos'] = True

    print("getting Dictionary...")
    if not np.isfortran(D):
        D=np.asfortranarray(D)
    alphas = spams.lasso(embedding, D=D, **lasso_params)
    eval_alphas = spams.lasso(eval_embedding, D=D, **lasso_params)

    print("getting labels...")
    labels_to_vecs,labels_to_ids,_,_ = get_label_mappings(alphas.T, training_data, id_to_gold)

    mtx = scipy.sparse.vstack([labels_to_vecs[row] for row in sorted(labels_to_vecs)])

    MA = get_pmis(mtx)

    R = eval_alphas.T
  
    predictions = {}
    ids, preds = [], []

    print("evaluate dataset:")

    sense_inventory_a, sense_inventory_n, sense_inventory_v, sense_inventory_r  = get_sense_inventory()


    idx = 0
    for sentence in tqdm(test_data):
        for word in sentence["words"]:
            vec = R[idx]

            possible_indices = range(len(labels_to_ids))

            token_id = word['id']
            lemma = word ['lemma']

            pos = word['pos']
            if pos.lower()[0] == "v" and lemma in sense_inventory_v.keys():
                candidates = sense_inventory_v[lemma]
            elif pos.lower() == "adv" and lemma in sense_inventory_r.keys():
                candidates = sense_inventory_r[lemma]
            elif pos.lower()[0] == "n" and lemma in sense_inventory_n.keys():
                candidates = sense_inventory_n[lemma]
            elif pos.lower() == "adj" and lemma in sense_inventory_a.keys():
                candidates = sense_inventory_a[lemma]
            else:
                candidates = []

            potential_senses = []
            #potential_synsets = []
            #potential_lexnames = []

            for candidate in candidates:
                #synset_name = synset_lexname = candidate
                #synset = wn.lemma_from_key(candidate).synset()
                #synset_name = synset.name()
                #synset_lexname = synset.lexname()
                potential_senses.append(candidate)
                #potential_synsets.append(synset_name)
                #potential_lexnames.append(synset_lexname)


            synset_indices = [labels_to_ids[s] if s in labels_to_ids else -1 for s in potential_senses]

            if len(synset_indices)==0:
                continue
            else:
                possible_indices = synset_indices

            ids.append(token_id)
            scores = MA[possible_indices] @ vec.T
            preds.append(potential_senses[np.argmax(scores)])
            predictions[token_id] = set([preds[-1]])
            idx+=1

    with open(path, 'w') as f:
        for lemma_id, pred in zip(ids, preds):
            f.write('{} {}\n'.format(lemma_id, pred))






