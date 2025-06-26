import argparse
import os
import json
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk import download
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
import spacy
from gensim import corpora, models

download('punkt')
MAX_TOKENS = 128

# === truncate ===
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

# === Основная функция ===
def main(args):
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in tqdm(f, desc="Загрузка JSON")]
    df = pd.DataFrame(data)
    df = df[['user_id', 'business_id', 'text']]

    # === Инициализация моделей ===
    sbert = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')
    # sentiment_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=0)
    # Загрузка модели
    sent_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    sent_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    sent_model = sent_model.to('cuda')
    sent_model.eval()
    nlp = spacy.load("en_core_web_sm")

    # === Gensim LDA (по всей коллекции) ===
    tokenized = [word_tokenize(truncate_text(t)) for t in df['text']]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=10, passes=5)

    # === DataLoader ===
    dataset = ReviewDataset(df)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)

    # Хранилища
    user_embeddings, user_sentiments, user_topics, user_entities = {}, {}, {}, {}
    biz_embeddings, biz_sentiments, biz_topics, biz_entities = {}, {}, {}, {}
    user_counts = {}
    biz_counts = {}

    for texts, users, businesses in tqdm(loader, desc="Обработка батчей"):
        embs = sbert.encode(texts, convert_to_numpy=True, batch_size=args.batch_size, device='cuda')
        # sentiments = sentiment_pipe(texts, truncation=True, max_length=MAX_TOKENS)

        with torch.no_grad():
            batch = sent_tokenizer(texts, padding=True, truncation=True, max_length=MAX_TOKENS,
                                return_tensors='pt').to('cuda')
            outputs = sent_model(**batch)
            probs = F.softmax(outputs.logits, dim=1)
            sentiment_scores = (probs[:, 2] - probs[:, 0]).cpu().numpy()  # POS - NEG

        for i, text in enumerate(texts):
            uid, bid = users[i], businesses[i]
            emb = embs[i]
            sent_score = sentiment_scores[i]
            # sent_label = sentiments[i]['label']
            # sent_score = sentiments[i]['score'] if sent_label == 'POSITIVE' else -sentiments[i]['score']

            # Топики
            bow = dictionary.doc2bow(word_tokenize(text))
            topic_dist = np.zeros(10)
            for topic_id, weight in lda.get_document_topics(bow):
                topic_dist[topic_id] = weight

            # NER
            doc = nlp(text)
            entities = [ent.label_ for ent in doc.ents]

            for tgt, E, S, T, N, C in [(uid, user_embeddings, user_sentiments, user_topics, user_entities, user_counts),
                                    (bid, biz_embeddings, biz_sentiments, biz_topics, biz_entities, biz_counts)]:
                E.setdefault(tgt, []).append(emb)
                S.setdefault(tgt, []).append(sent_score)
                T.setdefault(tgt, []).append(topic_dist)
                N.setdefault(tgt, []).extend(entities)
                C[tgt] = C.get(tgt, 0) + 1

    def agg_entities(entity_list):
        from collections import Counter
        counter = Counter(entity_list)
        total = sum(counter.values())
        return np.array([counter.get(label, 0)/total for label in ['ORG', 'LOC', 'PERSON', 'PRODUCT']])

    # === Агрегация и сохранение ===
    with h5py.File(args.output_hdf5, 'w') as f:
        for name, E, S, T, N, C in [
            ('user', user_embeddings, user_sentiments, user_topics, user_entities, user_counts),
            ('business', biz_embeddings, biz_sentiments, biz_topics, biz_entities, biz_counts)
        ]:
            group = f.create_group(name)
            for key in tqdm(E.keys(), desc=f"Агрегация {name}"):
                emb = average_vectors(E[key])
                sent = np.mean(S[key]) if S[key] else 0.0
                topic = average_vectors(T[key])
                entity_vec = agg_entities(N[key]) if N[key] else np.zeros(4)
                count = C.get(key, 1)
                full_vec = np.concatenate([emb, topic, entity_vec, [sent, count]])
                group.create_dataset(key, data=full_vec, compression="gzip")

    print(f"\n✅ Всё сохранено в: {args.output_hdf5}")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, required=True)
    parser.add_argument('--output_hdf5', type=str, default='embeddings_enriched.hdf5')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    main(args)
