# data_processor.py
import pandas as pd
import numpy as np
import json
import ast
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'yelp_data'
OUTPUT_DIR = 'processed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_nested_attributes(attributes):
    attributes_parsed = pd.json_normalize(attributes)
    
    # колонки с вложенными словарями
    nested_cols = []
    for col in attributes_parsed.columns:
        
        if attributes_parsed[col][attributes_parsed[col].notna()].astype(str).str.contains('{').any():
            nested_cols.append(col)

    for col in nested_cols:
        notna_attributes = attributes_parsed[col][attributes_parsed[col].notna()]
        
        if not notna_attributes.empty:
            # переводим str в dict
            nested_attributes = notna_attributes.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
            # новый df из вложенных словарей
            normalized_col_df = pd.json_normalize(nested_attributes)
            normalized_col_df.columns = [f'{col}_{orig_col}' for orig_col in normalized_col_df.columns]
            
            attributes_parsed = attributes_parsed.drop(col, axis=1)
            attributes_parsed = pd.concat([attributes_parsed, normalized_col_df], axis=1)
            
    return attributes_parsed


def clean_attribute_values(attributes_df):
    df = attributes_df.copy()
    
    for col in df.columns:
        df[col] = df[col].astype(str).str.extract(r"u?'?([^']*)'?", expand=False)
        replacements = {'True': True, 'true': True, 'False': False, 'false': False, 
                        'nan': np.nan, 'none': np.nan, 'NaN': np.nan, 'None': np.nan, 'null': np.nan, '': np.nan}
        df[col] = df[col].replace(replacements)
        
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    return df


def encode_attributes(attributes_df):
    df = attributes_df.copy()
    ordinal_maps = {
        'Alcohol': {'nan': 0, 'none': 0, 'beer_and_wine': 1, 'full_bar': 2},
        'RestaurantsAttire': {'nan': 0, 'casual': 1, 'dressy': 2, 'formal': 3},
        'NoiseLevel': {'nan': 0, 'quiet': 1, 'average': 2, 'loud': 3, 'very_loud': 4},
        'Smoking': {'nan': 0, 'no': 1, 'outdoor': 2, 'yes': 3}, 
        'AgesAllowed': {'nan': 0, 'allages': 1, '18plus': 2, '21plus': 3}
    }
    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping)

    nominal_cols = ['WiFi', 'BYOBCorkage']
    for col in nominal_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            try:
                df[col] = df[col].astype(bool).astype(int)
            except Exception:
                pass
    
    if 'BusinessAcceptsCreditCards' in df.columns:
        df.BusinessAcceptsCreditCards = df.BusinessAcceptsCreditCards.fillna(1)

    df = df.fillna(0)
    return df


def process_business_data(business_json_path):
    print(f'Обработка {business_json_path}...')
    business_df = pd.read_json(business_json_path, lines=True)

    business_df = business_df[business_df.is_open == 1].reset_index(drop=True)
    print(f'Всего строк: {len(business_df)}')
    
    # атрибуты
    attributes_series = business_df['attributes'].fillna({}) 
    attributes_parsed = parse_nested_attributes(attributes_series)
    attributes_cleaned = clean_attribute_values(attributes_parsed)
    attributes_encoded = encode_attributes(attributes_cleaned)
    
    business_df = business_df.drop(columns=['attributes'])
    business_df = pd.concat([business_df, attributes_encoded], axis=1)

    # категории
    business_categories_list = []
    categories = business_df['categories'].str.split(', ').apply(
        lambda x: [cat.lower().strip().replace('/', ' or ').replace('&', ' and ')  for cat in x if isinstance(cat, str)] if isinstance(x, list) else []
    )

    for business_id, cats in zip(business_df['business_id'], categories):
        if isinstance(cats, list):
            for cat in cats:
                business_categories_list.append({'business_id': business_id, 'category': cat})
    
    has_category_df = pd.DataFrame(business_categories_list)
    
    print(f'Business-category отношений: {len(has_category_df)}')
    has_category_df.to_csv(os.path.join(OUTPUT_DIR, 'business_has_category.csv'), index=False)
    
    unique_categories = pd.DataFrame({'category_name': has_category_df['category'].unique()})
    unique_categories.to_csv(os.path.join(OUTPUT_DIR, 'categories.csv'), index=False)
    print(f'Уникальных категорий: {len(unique_categories)}')

    # ненужные столбцы
    cols_to_drop = ['is_open', 'hours', 'name', 'address', 'state', 'postal_code', 'categories']
    cols_to_drop = [col for col in cols_to_drop if col in business_df.columns]
    
    final_business_df = business_df.drop(columns=cols_to_drop)
    final_business_df.to_csv(os.path.join(OUTPUT_DIR, 'businesses_cleaned.csv'), index=False)
    print(f'Очищенные данные сохранены в: {os.path.join(OUTPUT_DIR, 'businesses_cleaned.csv')}')
    
    return final_business_df, has_category_df


def process_user_data(user_json_path):
    print(f'Обработка {user_json_path}...')

    with open(user_json_path, 'r', encoding='utf-8') as f:
        user_df = pd.DataFrame([json.loads(line) for line in tqdm(f, desc="Загрузка JSON")])
    
    print(f'Всего юзеров: {len(user_df)}')

    # friends
    user_friends_list = []
    
    friends = user_df['friends'].replace('None', '').str.split(', ')
        
    for user_id, friend_ids_list in zip(user_df['user_id'], friends):
        if isinstance(friend_ids_list, list):
            for friend_id in friend_ids_list:
                friend_id = friend_id.strip()
                if friend_id:
                    user_friends_list.append({'user_id_from': user_id, 'user_id_to': friend_id})
    
    user_friends_df = pd.DataFrame(user_friends_list)
    
    print(f'User-user отношений: {len(user_friends_df)}')
    user_friends_df.to_csv(os.path.join(OUTPUT_DIR, 'user_friends_edges.csv'), index=False)

    # тут можно и другие колонки попробовать взять
    cols_to_keep = ['user_id', 'name', 'review_count', 'yelping_since', 'useful', 'funny', 'cool', 'fans', 'average_stars']
    cols_to_keep = [col for col in cols_to_keep if col in user_df.columns]

    final_user_df = user_df[cols_to_keep]
    final_user_df.to_csv(os.path.join(OUTPUT_DIR, 'users_cleaned.csv'), index=False)

    print(f'Очищенные данные сохранены в: {os.path.join(OUTPUT_DIR, 'users_cleaned.csv')}')
    
    return final_user_df, user_friends_df


def process_review_data(review_json_path):
    print(f'Обработка {review_json_path}...')
    
    reviews_data = []
    with open(review_json_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Загрузка JSON'):
            review = json.loads(line)
            # тут не нужны все колонки
            reviews_data.append({
                'review_id': review.get('review_id'),
                'user_id': review.get('user_id'),
                'business_id': review.get('business_id'),
                'stars': review.get('stars'),
                'date': review.get('date'),
                'text_length': len(review.get('text', ''))
            })
    
    reviews_df = pd.DataFrame(reviews_data)
    reviews_df.to_csv(os.path.join(OUTPUT_DIR, 'review_interactions.csv'), index=False)
    print(f'Данные сохранены в: {os.path.join(OUTPUT_DIR, 'review_interactions.csv')}')
    
    return reviews_df

if __name__ == '__main__':
    
    business_df, has_category_df = process_business_data(os.path.join(DATA_DIR, 'yelp_academic_dataset_business.json'))
    user_df, user_friends_df = process_user_data(os.path.join(DATA_DIR, 'yelp_academic_dataset_user.json'))

    # использовать уже разделенные train/test файлы ????????????????????????????????????
    # process_review_data('train.json') 
    # process_review_data('test.json')
    reviews_df = process_review_data(os.path.join(DATA_DIR, 'yelp_academic_dataset_review.json'))

    # checkin.json
    checkin_df = pd.read_json(os.path.join(DATA_DIR, 'yelp_academic_dataset_checkin.json'), lines=True)
    checkin_df.to_csv(os.path.join(OUTPUT_DIR, 'checkins.csv'), index=False)
    
    print('\n--- Phase 1 Complete ---')
    print(f'Файлы сохранены в: {OUTPUT_DIR}')