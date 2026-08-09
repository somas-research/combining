# Evaluating Latent Semantic Pre-training for Fine-grained Word Sense Disambiguation

This repository contains the code accompanying the paper:

> **Making Sense of Pre-training: Evaluating Masked Latent Semantic Modeling for Word Sense Disambiguation**

The project investigates the effect of **Masked Latent Semantic Modeling (MLSM)** pretraining on unsupervised Word Sense Disambiguation (WSD) methods based on sparse semantic representations derived through dictionary learning.

The repository currently supports:
- centroid-based semantic matching,
- PMI-based sparse semantic matching,
- layer-wise probing experiments,
- dictionary learning–based semantic profiles.

---

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Required Resources

The experiments require:
- the `WSD_Evaluation_Framework`,
- the `wsd_hard_benchmark` datasets.

After downloading the required resources, ensure that the paths specified in the configuration files correctly point to the local directories.

---

## Running Experiments

Experiments are executed through:

```bash
python main.py --config configs/example.json
```

---

## Configuration

Typical configuration options include:

- `models`: pretrained checkpoints to evaluate,
- `eval_datasets`: evaluation datasets,
- `layers`: transformer layers used for probing,
- `method`: `centroid` or `pmi`,
- `K`: dictionary size,
- `lda`: sparsity regularization parameter,
- `batchsize`: number of token representations processed per batch during dictionary learning,
- `iters`: dictionary learning iterations.

Example configuration:

```json
{
  "examples": true,
  "wngc": true,
  "method": "centroid",
  "models": [
    "path/to/model"
  ],
  "eval_datasets": [
    "path/to/dataset"
  ],
  "layers": [-4, -3, -2, -1],
  "K": 3000,
  "batchsize": 400,
  "lda": 0.05
}
```

