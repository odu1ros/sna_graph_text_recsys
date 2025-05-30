import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import os
import csv
import h5py
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from collections import defaultdict

OUTPUT_DIR = 'processed_data'
FEATURES_DIR = 'storage'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
FEATURES_HDF5_PATH = os.path.join(FEATURES_DIR, 'business_category_features.hdf5')

def generate_categories_embeddings(has_category_df,
                                   sbert_model_name=SBERT_MODEL_NAME,
                                   device=DEVICE,
                                   batch_size=128):
    
    print('Генерация эмбеддингов категорий...')
    sbert = SentenceTransformer(sbert_model_name).to(device)
    
    unique_categories = sorted(list(has_category_df['category'].astype(str).unique()))

    categories_embeddings = {}

    embeddings_raw = sbert.encode(unique_categories, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)

    for cat, emb in zip(unique_categories, embeddings_raw):
        categories_embeddings[cat] = emb

    embedding_dim = sbert.get_sentence_embedding_dimension()

    return categories_embeddings, embedding_dim

def aggregate_category_embeddings(businesses_df, has_category_df, categories_embeddings, embedding_dim):
    
    business_agg_category_embeddings = []
    # список как в начальном датасете
    business_has_categories = has_category_df.groupby('business_id')['category'].apply(list)

    for business_id in tqdm(businesses_df['business_id']):
        
        cur_business_categories = business_has_categories.get(business_id, [])
        cur_business_categories_embeddings = []

        # эмбеддинги каждой категории бизнеса
        for cat in cur_business_categories:
            cat_emb = categories_embeddings.get(str(cat))
            if cat_emb is not None:
                cur_business_categories_embeddings.append(cat_emb)
        
        # усредненный
        if cur_business_categories_embeddings:
            avg_embedding = np.mean(cur_business_categories_embeddings, axis=0)
        else:
            avg_embedding = np.zeros(embedding_dim if embedding_dim > 0 else 1)
            
        business_agg_category_embeddings.append(avg_embedding)

    return np.array(business_agg_category_embeddings)


def save_business_category_features_hdf5(businesses_df, business_category_embeddings, output_path=FEATURES_HDF5_PATH):
    with h5py.File(output_path, 'w') as f:
        # если нужно эти фичи КОНКАТЕНИРОВАТЬ с фичами для узла 'business',
        # то потом в graph_builder.py прочитать нлпшные и эти и соединить
        
        group = f.create_group('business_category_embeddings')
        
        business_ids = businesses_df['business_id'].tolist()
        
        for i, business_id in enumerate(tqdm(business_ids, desc='Сохранение эмбеддингов business-category')):
            # можно добавить метаданные типа количество категорий у бизнеса
            vec = business_category_embeddings[i]
            group.create_dataset(business_id, data=vec, compression='gzip')
    print('Эмбеддинги business-category сохранены')


def save_category_embeddings_hdf5(categories_embeddings, output_path=FEATURES_HDF5_PATH):
    with h5py.File(output_path, 'w') as f:
        group = f.create_group('category_embeddings')
        
        for cat, vec in tqdm(categories_embeddings.items(), desc='Сохранение эмбеддингов названий категорий'):
            group.create_dataset(str(cat), data=vec, compression="gzip")
                
    print(f'Эмбеддинги названий категорий сохранены')


def create_hnsw_index_for_faiss(vec, M=32, ef_construction=100, normalize=True):
    
    d = vec.shape[1]
    vec_copy = np.ascontiguousarray(vec.astype(np.float32))
    
    if normalize:
        faiss.normalize_L2(vec_copy)
    
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT if normalize else faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.add(vec_copy)

    return index

def update_scores_dict(current_scores_dict,
                       indices_array,       # feature_indices[0]
                       distances_array,     # feature_distances[0]
                       feature,             # 'cat', 'attr', 'loc'
                       current_business_idx):
    # current_scores_dict это словарь {neighbor_idx: {'cat': similarity, 'attr': similarity, 'loc': similarity}}, число соседей = k
    for k_idx, neighbor_idx in enumerate(indices_array):
        if neighbor_idx == current_business_idx or neighbor_idx == -1:
            continue # пропускаем себя и невалидные индексы (если не нашлось k соседей)
        current_scores_dict[neighbor_idx][feature] = distances_array[k_idx]

def get_business_similarities_faiss(
    business_df, 
    business_category_embeddings,
    idx_to_business_id,
    K_neighbors=20,
    category_weight=0.4,
    attribute_weight=0.4,
    location_weight=0.2,
    similarity_threshold=0.3
):
    
    # категории
    print('Строим фичи по категориям...')

    categories = business_category_embeddings
    category_index_faiss = create_hnsw_index_for_faiss(categories, normalize=True)

    # атрибуты
    print('Строим фичи по атрибутам...')

    skip_attributes = ['business_id', 'name', 'address', 'city', 'state', 'postal_code', 
                       'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'categories', 'hours']
    attribute_cols = [col for col in business_df.columns if col not in skip_attributes] 
    
    attributes = business_df[attribute_cols].to_numpy()
    
    scaler = StandardScaler()
    attributes_scaled = scaler.fit_transform(attributes)
    attribute_index_faiss = create_hnsw_index_for_faiss(attributes_scaled, normalize=True)

    # location
    print('Строим фичи по местоположению...')
    coords = business_df[['latitude', 'longitude']].to_numpy().astype(np.float32)
    # здесь по-хорошему надо бы по haversine считать, а не по l2, но тогда придется какие-нибудь kd деревья пилить с кастомной метрикой
    location_index_faiss = create_hnsw_index_for_faiss(coords, normalize=False) 

    
    potential_edges = defaultdict(float) # {(idx1, idx2): combined_score, ...}

    N = len(business_df)

    print(f'Поиск K={K_neighbors} соседей и подсчет схожестей...')
    for i in tqdm(range(N)):
        # категории
        cat_vec = np.ascontiguousarray(categories[i:i+1].astype(np.float32)) # shape: (1, embedding_dim)
        faiss.normalize_L2(cat_vec)
        cat_distances, cat_indices = category_index_faiss.search(cat_vec, K_neighbors + 1)
        
        # атрибуты
        attr_vec = np.ascontiguousarray(attributes_scaled[i:i+1].astype(np.float32))
        faiss.normalize_L2(attr_vec)
        attr_distances, attr_indices = attribute_index_faiss.search(attr_vec, K_neighbors + 1)

        # локация
        loc_vec = np.ascontiguousarray(coords[i:i+1].astype(np.float32))
        loc_distances, loc_indices = location_index_faiss.search(loc_vec, K_neighbors + 1)
        
        # rbf ядром преобразуем расстояние в схожесть, чтобы 0 было плохо 1 хорошо
        # сигму не очень понятно как оценивать корректно, я напишу локальное стд
        # возможно лучше просто обратное расстояние брать
        sigma = np.std(np.sqrt(loc_distances[0]))**2

        if sigma < 1e-6:
            sigma = 1.0

        loc_similarities = np.exp(-loc_distances[0] / (2 * sigma))

        
        current_neighbors_scores = defaultdict(lambda: {'cat': 0.0, 'attr': 0.0, 'loc': 0.0})

        update_scores_dict(current_neighbors_scores, cat_indices[0], cat_distances[0], 'cat', i)
        update_scores_dict(current_neighbors_scores, attr_indices[0], attr_distances[0], 'attr', i)
        update_scores_dict(current_neighbors_scores, loc_indices[0], loc_similarities, 'loc', i)

        # взвешиваем скоры
        for neighbor_idx, scores in current_neighbors_scores.items():
            combined_score = (category_weight * scores['cat'] +
                              attribute_weight * scores['attr'] +
                              location_weight * scores['loc'])
            
            # ребра ненаправленные
            u, v = min(i, neighbor_idx), max(i, neighbor_idx)

            if combined_score > similarity_threshold:
                 # на случай, если бизнесу А попал в соседи бизнес Б, а бизнесу Б бизнес А - нет, возьмем макс схожесть
                 # можно в теории делать направленный граф, но думаю так норм
                 potential_edges[(u,v)] = max(potential_edges.get((u,v), 0.0), combined_score)


    final_edges = []
    for (u, v), score in potential_edges.items():
        final_edges.append((idx_to_business_id[u], idx_to_business_id[v], score)) # [(business_id1, business_id2, combined_score), ...]

    print(f'Отобрано business-business ребер: {len(final_edges)}')
    
    return final_edges


def edges_to_csv(edges, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['business_id_u', 'business_id_v', 'similarity_score'])
        writer.writerows(edges)
    print(f'В {output_path} сохранено business-business ребер: {len(edges)}')


if __name__ == "__main__":
    print(f'device: {DEVICE}')
    businesses_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'businesses_cleaned.csv'))
    has_category_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'business_has_category.csv'))

    CATEGORIES_EMBEDDINGS_HDF5_PATH = os.path.join(FEATURES_DIR, 'category_name_embeddings.hdf5')
    BUSINESS_CATEGORY_EMBEDDINGS_HDF5_PATH = os.path.join(FEATURES_DIR, 'business_category_features.hdf5')

    # эмбеддинги для названий категорий
    categories_embeddings, embedding_dim = generate_categories_embeddings(has_category_df)
    save_category_embeddings_hdf5(categories_embeddings, CATEGORIES_EMBEDDINGS_HDF5_PATH)

    # агрегация эмбеддингов категорий для каждого бизнеса
    business_category_embeddings = aggregate_category_embeddings(businesses_df, has_category_df, categories_embeddings, embedding_dim)
    print(f'Размерность business-category эмбеддингов: {business_category_embeddings.shape}')
    save_business_category_features_hdf5(businesses_df, business_category_embeddings, BUSINESS_CATEGORY_EMBEDDINGS_HDF5_PATH)

    # схожести бизнесов
    business_ids_unique = businesses_df['business_id'].unique()
    idx_to_business_id = {idx: business_id for idx, business_id in enumerate(business_ids_unique)}

    print('\n Расчет схожестей бизнесов')
    business_similarities = get_business_similarities_faiss(
        businesses_df,
        business_category_embeddings,
        idx_to_business_id,
        K_neighbors=30,
        similarity_threshold=0.5
    )
    edges_to_csv(business_similarities, os.path.join(OUTPUT_DIR, 'business_similarities.csv'))

    print('\n --- Признаки сгенерированы ---')