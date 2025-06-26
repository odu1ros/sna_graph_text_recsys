extract yelp dataset

```
mkdir yelp_dataset
tar -xf yelp_dataset.tar -C yelp_dataset
```

process jsons and get csvs (output_folder: processed_data)

required: yelp dataset in dir: yelp_data/

```python preprocess_business.py```

get embeddings (categories, business-categories; output_folder: storage), business similarities (output_folder: processed_data)

```python business_similarities.py```

Prepare embeddings (2021 - train, 2022 - validation and test):

```
sh train_test_split.sh yelp_dataset/yelp_academic_dataset_review.json
sh process_reviews.sh train.json
mv storage/embeddings_chunk_00.hdf5 storage/embeddings_2021.hdf5
sh process_reviews.sh test.json
mv storage/embeddings_chunk_00.hdf5 storage/embeddings_2022.hdf5
```

get HeteroData graph (output_folder: processed_data)

required: storage/embeddings_2022_full.hdf5

required: train.json

```python build_graph.py```

run train test (cold start, se embeddings as of 2022)

```python train_cold_start.py --graph_path processed_data/hetero_graph_v1_train.pt --train_json train.json --val_json test.json --lr 0.0001 --epochs 250```

