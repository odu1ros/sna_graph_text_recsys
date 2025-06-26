import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import h5py
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score


class LightGCNEncoder(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers=3, pretrained_user=None, pretrained_item=None):
        super().__init__()
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.reset_parameters()

        if pretrained_user is not None:
            self.user_embedding.weight.data.copy_(pretrained_user)
        if pretrained_item is not None:
            self.item_embedding.weight.data.copy_(pretrained_item)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index, edge_weight):
        src, dst = edge_index
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight

        all_user_embs = [u_emb]
        all_item_embs = [i_emb]

        for _ in range(self.num_layers):
            deg_u = torch.bincount(src, minlength=u_emb.size(0)).float().clamp(min=1)
            deg_i = torch.bincount(dst, minlength=i_emb.size(0)).float().clamp(min=1)
            norm = 1.0 / torch.sqrt(deg_u[src] * deg_i[dst])

            weighted_ui = i_emb[dst] * (norm * edge_weight).unsqueeze(1)
            weighted_iu = u_emb[src] * (norm * edge_weight).unsqueeze(1)

            agg_user = torch.zeros_like(u_emb).index_add(0, src, weighted_ui)
            agg_item = torch.zeros_like(i_emb).index_add(0, dst, weighted_iu)

            u_emb = agg_user
            i_emb = agg_item

            all_user_embs.append(u_emb)
            all_item_embs.append(i_emb)

        final_user = torch.stack(all_user_embs, dim=1).mean(dim=1)
        final_item = torch.stack(all_item_embs, dim=1).mean(dim=1)
        return final_user, final_item

    def l2_regularization(self):
        return 0.5 * (self.user_embedding.weight.norm(2).pow(2) + self.item_embedding.weight.norm(2).pow(2))


class EdgeDecoder(nn.Module):
    def forward(self, z_user, z_item, edge_index):
        src, dst = edge_index
        return (z_user[src] * z_item[dst]).sum(dim=1)


def load_embeddings(hdf5_path, group):
    with h5py.File(hdf5_path, 'r') as f:
        group_data = f[group]
        ids = list(group_data.keys())
        vectors = [group_data[k][()] for k in ids]
    return ids, torch.tensor(np.stack(vectors), dtype=torch.float)


def build_index_mapping(ids):
    return {k: i for i, k in enumerate(ids)}


def parse_timestamp(ts):
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


def stream_edges_with_time(path, user_map, item_map, alpha=1e-3):
    edges = []
    weights = []
    timestamps = []
    max_time = None

    with open(path, 'r') as f:
        for line in f:
            r = json.loads(line)
            u, i, t = r['user_id'], r['business_id'], r['date']
            if u not in user_map or i not in item_map:
                continue
            timestamp = parse_timestamp(t)
            timestamps.append(timestamp)
            edges.append((user_map[u], item_map[i]))

    if not timestamps:
        raise ValueError("Нет допустимых рёбер")

    max_time = max(timestamps)
    for ts in timestamps:
        age = (max_time - ts).total_seconds() / 86400
        weights.append(np.exp(-alpha * age))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight


def sample_negative_edges(edge_index, num_items):
    src, _ = edge_index
    neg_dst = torch.randint(0, num_items, (src.size(0),), device=src.device)
    return src, neg_dst


def train(encoder, decoder, edge_index, edge_weight, optimizer, criterion, device, is_eval=False, lambda_reg=1e-5):
    if not is_eval:
        encoder.train()
        decoder.train()
        optimizer.zero_grad()
    else:
        encoder.eval()
        decoder.eval()

    user_emb, item_emb = encoder(edge_index.to(device), edge_weight.to(device))
    src, dst = edge_index.to(device)

    pos_logits = decoder(user_emb, item_emb, (src, dst))
    pos_labels = torch.ones_like(pos_logits)

    neg_src, neg_dst = sample_negative_edges(edge_index, item_emb.size(0))
    neg_logits = decoder(user_emb, item_emb, (neg_src.to(device), neg_dst))
    neg_labels = torch.zeros_like(neg_logits)

    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([pos_labels, neg_labels])
    loss = criterion(logits, labels)

    if hasattr(encoder, 'l2_regularization'):
        loss = loss + lambda_reg * encoder.l2_regularization()

    if not is_eval:
        loss.backward()
        optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(encoder, decoder, edge_index, edge_weight, device):
    encoder.eval()
    decoder.eval()
    user_emb, item_emb = encoder(edge_index.to(device), edge_weight.to(device))
    src, dst = edge_index.to(device)

    pos_logits = decoder(user_emb, item_emb, (src, dst)).sigmoid().cpu().numpy()
    pos_labels = np.ones_like(pos_logits)

    neg_src, neg_dst = sample_negative_edges(edge_index, item_emb.size(0))
    neg_logits = decoder(user_emb, item_emb, (neg_src.to(device), neg_dst)).sigmoid().cpu().numpy()
    neg_labels = np.zeros_like(neg_logits)

    logits = np.concatenate([pos_logits, neg_logits])
    labels = np.concatenate([pos_labels, neg_labels])
    return roc_auc_score(labels, logits), average_precision_score(labels, logits)


def load_ids(hdf5_path, group):
    with h5py.File(hdf5_path, 'r') as f:
        return list(f[group].keys())

def main(args):
    user_ids = load_ids(args.embeddings_train, 'user')
    item_ids = load_ids(args.embeddings_train, 'business')
    user_map = build_index_mapping(user_ids)
    item_map = build_index_mapping(item_ids)

    print("📥 Загрузка train...")
    train_edge, train_weight = stream_edges_with_time(args.train_json, user_map, item_map, alpha=args.alpha)
    print("📥 Загрузка val...")
    val_edge, val_weight = stream_edges_with_time(args.val_json, user_map, item_map, alpha=args.alpha)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = LightGCNEncoder(len(user_map), len(item_map), args.dim, args.num_layers).to(device)
    decoder = EdgeDecoder().to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    patience = 50
    counter = 0

    for epoch in range(1, args.epochs + 1):
        loss = train(encoder, decoder, train_edge, train_weight, optimizer, criterion, device, lambda_reg=args.lambda_reg)
        val_loss = train(encoder, decoder, val_edge, val_weight, None, criterion, device, is_eval=True)
        print(f"[Epoch {epoch:03d}] Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"🛑 Early stopping after {patience} epochs")
                break

    auc, ap = evaluate(encoder, decoder, val_edge, val_weight, device)
    print(f"✅ Final ROC-AUC: {auc:.4f} | AP: {ap:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_train', type=str, required=True)
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--val_json', type=str, required=True)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=1e-3)
    parser.add_argument('--lambda_reg', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)