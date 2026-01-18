from utils import *
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import centroid
import pmi
import os 

config = {
    "examples": True,
    "wngc": True,
    "method": "centroid",
    "models":  [
                "path/to/model1",
                "path/to/model2"
    ],
    "eval_datasets": [
                 "../wsd-hard-benchmark/wsd_hard_benchmark/softEN/softEN",
                 #"../wsd-hard-benchmark/wsd_hard_benchmark/hardEN/hardEN",
                 #"../wsd-hard-benchmark/wsd_hard_benchmark/42D/42D",
                 #"../wsd-hard-benchmark/wsd_hard_benchmark/S10amended/S10amended",
                 #"../wsd-hard-benchmark/wsd_hard_benchmark/ALLamended/ALLamended"
                 ],
    "train_path": "../WSD_Evaluation_Framework/Training_Corpora/",
    "layers": [-4, -3, -2, -1],
    "run_name": "wngc_ex",
    "iters": 100,
    "batchsize": 400,
    "lda": 0.05,
    "K": 3000
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id_to_gold, _ = get_labels(config["train_path"] + 'SemCor/semcor.gold.key.txt')
    training_data, _ = get_data(config["train_path"] + 'SemCor/semcor.data.xml', id_to_gold)

    training_data_wngc=None
    if config["wngc"]:
        id_to_gold_wngc, _ = get_labels(config["train_path"] + 'wngc/wngc.gold.key.txt')
        training_data_wngc, _ = get_data(config["train_path"] +'wngc/wngc.data.xml', id_to_gold_wngc)
        id_to_gold.update(id_to_gold_wngc)

        
    for model_path in config["models"]:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, output_hidden_states=True).to(device)

        if model_path.startswith("/data/"):
            model_name = model_path.split("/")[-2]
        else:
            model_name = model_path.split("/")[-1]


        embeddings = load_or_compute_embeddings(
            model_name,
            config["layers"],
            get_train_embeddings,
            (model, tokenizer, training_data)
        )
        sources = [embeddings]

        if(config["wngc"]):
            embeddings1 = load_or_compute_embeddings(
                model_name,
                config["layers"],
                get_train_embeddings,
                (model, tokenizer, training_data_wngc),
                prefix = "_wngc"
            )
            sources.append(embeddings1)

        wn_senses=None
        if(config["examples"]):
            wn_emb, wn_senses = load_or_compute_wn(model_name, config["layers"])
            sources.append(wn_emb)
        

        combined_embeddings = np.array([
            np.concatenate(items, axis=0)
            for items in zip(*sources)
        ])
        print(combined_embeddings.shape)

        for eval_set in config["eval_datasets"]:
            setname = eval_set.split("/")[-1]
            print(f"Processing {model_name} on {setname}")
            test_data,_ = get_data(eval_set + ".data.xml", train=False)
            if config["method"] == "centroid":
                for layer in config["layers"]:
                    output_name = f"./centroid_outputs/{setname}_{model_name}_{str(layer)}_{config['run_name']}.key"
                    if not os.path.exists("./centroid_outputs/"):
                        os.mkdir("./centroid_outputs/")
                    centroid.evaluate_to_file(combined_embeddings[layer], test_data, model, tokenizer, layer, training_data=[training_data,training_data_wngc,wn_senses], id_to_gold=id_to_gold, path=output_name)
            else:
                eval_embeddings= pmi.get_eval_embeddings(model, tokenizer, config["layers"], test_data)
                for layer in config["layers"]:
                    l_embeddings = preprocess_data(combined_embeddings[layer])
                    l_eval_embeddings = preprocess_data(eval_embeddings[layer])
                    output_name = f"./sparse_outputs/{setname}_{model_name}_{str(layer)}_{config['run_name']}.key"
                    if not os.path.exists("./sparse_outputs/"):
                        os.mkdir("./sparse_outputs/")
                    pmi.evaluate_to_file(l_embeddings, l_eval_embeddings, config["K"], config["iters"], config["lda"], config["batchsize"], path=output_name, training_data=[training_data,training_data_wngc,wn_senses], id_to_gold=id_to_gold, test_data=test_data)