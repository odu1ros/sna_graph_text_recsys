import pandas as pd
import torch
from torch_geometric.data import HeteroData
# from torch_geometric.transforms import ToUndirected
import json
import h5py
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

PROCESSED_DATA_DIR = 'processed_data'
FEATURES_DIR = 'storage'
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)


BUSINESS_USER_FEATURES_HDF5 = os.path.join(FEATURES_DIR, 'embeddings_2022_full.hdf5')

BUSINESS_CATEGORY_AGG_FEATURES_HDF5 = os.path.join(FEATURES_DIR, 'business_category_features.hdf5')
CATEGORY_NAME_EMBEDDINGS_HDF5 = os.path.join(FEATURES_DIR, 'category_name_embeddings.hdf5')

# признаки из tips.json -?

INTERACTIONS_TRAIN_JSON = 'train.json' 

# ребра из data_processor.py
USER_FRIENDS_EDGES_CSV = os.path.join(PROCESSED_DATA_DIR, 'user_friends_edges.csv')
SIMILAR_BUSINESS_EDGES_CSV = os.path.join(PROCESSED_DATA_DIR, 'business_similarities.csv')

# граф
OUTPUT_GRAPH_PT = os.path.join(PROCESSED_DATA_DIR, 'hetero_graph_v1_train.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SBERT_MODEL_NAME_FOR_CITY = 'all-MiniLM-L6-v2'


def load_features_from_hdf5(hdf5_path, group_name, ids_ordered_list, expected_dim=None):
    data = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        group = f[group_name]
        for id_in_file in group.keys():
            data[id_in_file] = group[id_in_file][()]

    feature_dim = expected_dim
    if expected_dim is None:
        valid_vec = next((vec for vec in data.values() if hasattr(vec, 'shape') and len(vec.shape) > 0), None)
        feature_dim = valid_vec.shape[0]

    features_list = []
    for entity_id in ids_ordered_list:
        cur_vec = data.get(entity_id)
        if cur_vec is not None and hasattr(cur_vec, 'shape') and cur_vec.shape == (feature_dim,):
            features_list.append(cur_vec)
        else:
            # если id не найден
            features_list.append(np.zeros(feature_dim, dtype=np.float32))
        
    return np.stack(features_list).astype(np.float32)


def load_features_from_dataframe(df, id_column, feature_columns, ordered_ids_list, fill_na_value=0, dtype=np.float32):

    actual_feature_columns = [col for col in feature_columns if col in df.columns]

    df_indexed = df.set_index(id_column)
    df_aligned = df_indexed.reindex(ordered_ids_list)[actual_feature_columns]

    features = df_aligned.fillna(fill_na_value).to_numpy(dtype=dtype)
    
    if features.ndim == 1 and len(ordered_ids_list) > 0:
        if features.shape[0] == len(ordered_ids_list):
            features = features.reshape(-1, 1)
            
    return features


def build_hetero_graph():
    print(f"--- Построение графа ---")
    print(f'Device: {DEVICE}')
    data = HeteroData()

    print('Загрузка CSV файлов...')
    users_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'users_cleaned.csv'))
    businesses_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'businesses_cleaned.csv'))
    categories_metadata_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'categories.csv'))
    business_has_cat_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'business_has_category.csv'))
    
    
    interactions_data = []
    skipped_lines = 0
    with open(INTERACTIONS_TRAIN_JSON, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            try:
                cur_data = json.loads(line)
                interactions_data.append(cur_data)
            except json.JSONDecodeError as e:
                print(f'Ошибка декодирования JSON в строке {i+1}: {e}')
                print(f'Проблемная строка: {line.strip()[:100]}...')
                skipped_lines += 1
                continue
    if skipped_lines > 0:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Пропущено {skipped_lines} строк из-за ошибок декодирования JSON.")

    interactions_df = pd.DataFrame(interactions_data)

    # маппинги id -> idx
    ordered_user_ids = sorted(users_df['user_id'].unique().tolist())
    user_id_to_idx = {uid: i for i, uid in enumerate(ordered_user_ids)}

    ordered_business_ids = sorted(businesses_df['business_id'].unique().tolist())
    business_id_to_idx = {bid: i for i, bid in enumerate(ordered_business_ids)}
    
    ordered_category_names = sorted(categories_metadata_df['category_name'].unique().tolist())
    category_name_to_idx = {cat: i for i, cat in enumerate(ordered_category_names)}

    # ПРИЗНАКИ УЗЛОВ
    print('Загрузка признаков узлов...')

    # user
    
    user_features_list = []

    # из ревьюс
    user_embeddings = load_features_from_hdf5(BUSINESS_USER_FEATURES_HDF5,
                                              'user',
                                              ordered_user_ids,
                                              expected_dim=400)
    user_features_list.append(user_embeddings)
    
    # мета
    user_meta_cols = ['average_stars', 'useful', 'funny', 'cool', 'fans']
    user_meta = load_features_from_dataframe(users_df, 'user_id', user_meta_cols, ordered_user_ids)
    user_features_list.append(user_meta)

    data['user'].x = torch.tensor(np.concatenate(user_features_list, axis=1), dtype=torch.float)
    data['user'].node_id_str = ordered_user_ids
    print(f'Размерность фичей для user: {data['user'].x.shape}')

    # business

    business_features_list = []
    
    # из ревьюс
    business_embeddings = load_features_from_hdf5(BUSINESS_USER_FEATURES_HDF5,
                                                  'business',
                                                  ordered_business_ids,
                                                  expected_dim=400)
    business_features_list.append(business_embeddings)

    # эмбеддинги бизнес-категория
    business_category_embeddings = load_features_from_hdf5(BUSINESS_CATEGORY_AGG_FEATURES_HDF5, 
                                                           'business_category_embeddings',
                                                            ordered_business_ids,
                                                            expected_dim=384)
    business_features_list.append(business_category_embeddings)
    
    # мета
    business_meta_cols = ['stars']
    business_meta = load_features_from_dataframe(businesses_df, 'business_id', business_meta_cols, ordered_business_ids)
    business_features_list.append(business_meta)

    # мета (атрибуты)
    attr_excluded_cols = ['business_id', 'latitude', 'longitude', 'stars', 'review_count', 'city']
    attr_cols = [col for col in businesses_df.columns if col not in attr_excluded_cols]
    business_attr_meta = load_features_from_dataframe(businesses_df, 'business_id', attr_cols, ordered_business_ids)
    scaler = StandardScaler()
    business_attr_meta = scaler.fit_transform(business_attr_meta)
    business_features_list.append(business_attr_meta)
    
    data['business'].x = torch.tensor(np.concatenate(business_features_list, axis=1), dtype=torch.float)
    data['business'].node_id_str = ordered_business_ids
    print(f'Размерность фичей для business: {data['business'].x.shape}')


    # категории
    categories_embeddings = load_features_from_hdf5(CATEGORY_NAME_EMBEDDINGS_HDF5, 'category_embeddings', ordered_category_names)
    data['category'].x = torch.tensor(categories_embeddings, dtype=torch.float)
    data['category'].node_id_str = ordered_category_names
    print(f'Размерность фичей для category: {data['category'].x.shape}')
    


    # РЕБРА
    print('Построение ребер...')

    # (user) --interacts_with--> (business)
    src_user, dst_business = [], []
    edge_attrs_interaction_stars = []
    
    print(f'Обработка {INTERACTIONS_TRAIN_JSON}...')
    for _, row in tqdm(interactions_df.iterrows(), total=len(interactions_df)):
        uid, bid = row['user_id'], row['business_id']
        if uid in user_id_to_idx and bid in business_id_to_idx:
            src_user.append(user_id_to_idx[uid])
            dst_business.append(business_id_to_idx[bid])
            edge_attrs_interaction_stars.append(row.get('stars', 3.0))

    data['user', 'interacts', 'business'].edge_index = torch.tensor([src_user, dst_business], dtype=torch.long)
    data['user', 'interacts', 'business'].edge_attr = torch.tensor(edge_attrs_interaction_stars, dtype=torch.float).unsqueeze(1)

    # обратные ребра
    data['business', 'rev_interacts', 'user'].edge_index = data['user', 'interacts', 'business'].edge_index[[1,0]]
    data['business', 'rev_interacts', 'user'].edge_attr = data['user', 'interacts', 'business'].edge_attr
    print(f'Добавлено user-business отношений: {len(src_user)}')

    # (business) --belongs_to--> (category)
    src_business, dst_category = [], []
    
    print('Обработка: business-category...')
    for _, row in tqdm(business_has_cat_df.iterrows(), total=len(business_has_cat_df)):
        bid, cat = row['business_id'], row['category']
        if bid in business_id_to_idx and cat in category_name_to_idx:
            src_business.append(business_id_to_idx[bid])
            dst_category.append(category_name_to_idx[cat])
    
    data['business', 'belongs_to', 'category'].edge_index = torch.tensor([src_business, dst_category], dtype=torch.long)
    data['category', 'category_of', 'business'].edge_index = data['business', 'belongs_to', 'category'].edge_index[[1,0]] # reverse    
    print(f'Добавлено business-category отношений: {len(src_business)}')

    # (business) --similar_to--> (business)
    similar_business = pd.read_csv(SIMILAR_BUSINESS_EDGES_CSV)
    src_business, dst_business = [], []
    similarity_edges = []
    
    print('Обработка: business-business схожесть...')
    for _, row in tqdm(similar_business.iterrows(), total=len(similar_business)):
        bid_u, bid_v, score = row['business_id_u'], row['business_id_v'], row['similarity_score']
        if bid_u in business_id_to_idx and bid_v in business_id_to_idx:
            src_business.append(business_id_to_idx[bid_u])
            dst_business.append(business_id_to_idx[bid_v])
            similarity_edges.append(score)
    
    data['business', 'similar_to', 'business'].edge_index = torch.tensor([src_business, dst_business], dtype=torch.long)
    data['business', 'similar_to', 'business'].edge_attr = torch.tensor(similarity_edges, dtype=torch.float).unsqueeze(1)
    print(f'Added {len(src_business)} business-similarity edges.')

    # (user) --friends_with--> (user)
    friends_df = pd.read_csv(USER_FRIENDS_EDGES_CSV)
    src_user, dst_user = [], []

    print('Обработка: user-user...')
    mask_from_exists = friends_df['user_id_from'].isin(user_id_to_idx.keys())
    mask_to_exists = friends_df['user_id_to'].isin(user_id_to_idx.keys())
    valid_friends_df = friends_df[mask_from_exists & mask_to_exists].copy()

    if not valid_friends_df.empty:
        src_user = valid_friends_df['user_id_from'].map(user_id_to_idx).to_numpy(dtype=np.int64)
        dst_user = valid_friends_df['user_id_to'].map(user_id_to_idx).to_numpy(dtype=np.int64)

    edge_friends = torch.tensor(np.vstack((src_user, dst_user)), dtype=torch.long)
    data['user', 'friends_with', 'user'].edge_index = edge_friends
    print(f"Added {len(src_user)} user-friend edges.")
        
    # data = ToUndirected(merge=True)(data)
    
    data = data.to(DEVICE)

    print(f'Сохранение графа в: {OUTPUT_GRAPH_PT}...')
    torch.save(data, OUTPUT_GRAPH_PT)
    
    print("\n--- Построение графа завершено ---")
    print(data)
    for node_type, store in data.node_items():
        print(f'Узел: {node_type}, число узлов: {store.num_nodes}, размерность признаков: {store.x.shape}')
    for edge_type, store in data.edge_items():
        print(f'Ребро: {edge_type}, число ребер: {store.num_edges}, размерность признаков: {store.edge_index.shape}')

    return data

if __name__ == '__main__':
    
    final_graph = build_hetero_graph()
    print(f'\nГраф построен и сохранен в: {OUTPUT_GRAPH_PT}')