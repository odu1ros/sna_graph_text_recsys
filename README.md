process jsons and get csvs (output_folder: processed_data)
required: yelp dataset in dir: yelp_data/

```python preprocess_business.py```

get embeddings (categories, business-categories; output_folder: storage), business similarities (output_folder: processed_data)

```python business_similarities.py```

get HeteroData graph (output_folder: processed_data)
required: storage/embeddings_2022_full.hdf5

```python build_graph.py```

run train test

```python train_cold_start.py --graph_path processed_data/hetero_graph_v1_train.pt --train_json train.json --val_json test.json --lr 0.0001 --epochs 250```
