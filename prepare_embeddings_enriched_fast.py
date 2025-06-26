import argparse
import json
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# === Настройки ===
MAX_TOKENS = 128

# === Упрощённая обрезка текста ===
def truncate_text(text, max_tokens=MAX_TOKENS):
    tokens = word_tokenize(text)
    return ' '.join(tokens[:max_tokens])

# === Dataset ===
class ReviewDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe['text'].apply(truncate_text).tolist()
        self.user_ids = dataframe['user_id'].tolist()
        self.business_ids = dataframe['business_id'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'user_id': self.user_ids[idx],
            'business_id': self.business_ids[idx]
        }

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    user_ids = [item['user_id'] for item in batch]
    business_ids = [item['business_id'] for item in batch]
    return texts, user_ids, business_ids

# === Средняя агрегация ===
def average_vectors(vecs):
    return np.mean(vecs, axis=0) if vecs else np.zeros(len(vecs[0]))

def main(args):
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in tqdm(f, desc="Загрузка JSON")]
    df = pd.DataFrame(data)
    df = df[['user_id', 'business_id', 'text']]

    sbert = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')
    sia = SentimentIntensityAnalyzer()

    dataset = ReviewDataset(df)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    user_embeddings, user_sentiments = {}, {}
    biz_embeddings, biz_sentiments = {}, {}
    user_counts = {}
    biz_counts = {}
    for texts, users, businesses in tqdm(loader, desc="Обработка батчей"):
        embs = sbert.encode(texts, convert_to_numpy=True, batch_size=args.batch_size, device='cuda')
        sentiments = [sia.polarity_scores(text)['compound'] for text in texts]

        for i in range(len(texts)):
            uid, bid = users[i], businesses[i]
            emb = embs[i]
            sent = sentiments[i]

            for tgt, E, S, C in [
                (uid, user_embeddings, user_sentiments, user_counts),
                (bid, biz_embeddings, biz_sentiments, biz_counts)
            ]:
                E.setdefault(tgt, []).append(emb)
                S.setdefault(tgt, []).append(sent)
                C[tgt] = C.get(tgt, 0) + 1  # добавляем счётчик
    with h5py.File(args.output_hdf5, 'w') as f:
        for name, E, S, C in [
            ('user', user_embeddings, user_sentiments, user_counts),
            ('business', biz_embeddings, biz_sentiments, biz_counts)
        ]:
            group = f.create_group(name)
            for key in tqdm(E.keys(), desc=f"Агрегация {name}"):
                emb = average_vectors(E[key])
                sent = np.mean(S[key]) if S[key] else 0.0
                count = C.get(key, 1)
                full_vec = np.concatenate([emb, [sent, count]])
                group.create_dataset(key, data=full_vec, compression="gzip")

    print(f"\n✅ Эмбеддинги сохранены в: {args.output_hdf5}")

if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    nltk.download('vader_lexicon')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, required=True)
    parser.add_argument('--output_hdf5', type=str, default='embeddings_fast.hdf5')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    main(args)
