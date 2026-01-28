#!/bin/bash

python main.py \
    data=small \
    preprocessing=default \
    drug_embeddings=morgan \
    model=model1 \
    cold_start=kmedoids \
    drug_embeddings.use_chirality=true \
    preprocessing.max_isotonic_error=0.2 \
    model.n_layers=6