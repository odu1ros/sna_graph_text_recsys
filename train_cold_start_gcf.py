import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
import h5py
import json
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score


class NGCFEncoder(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.linear_layers_W1 = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        self.linear_layers_W2 = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for layer in self.linear_layers_W1:
            nn.init.xavier_uniform_(layer.weight)
        for layer in self.linear_layers_W2:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, edge_index):
        src, dst = edge_index  # user -> item

        u_emb = self.user_embedding.weight  # [num_users, dim]
        i_emb = self.item_embedding.weight  # [num_items, dim]

        all_user_embs = [u_emb]
        all_item_embs = [i_emb]

        for layer in range(self.num_layers):
            # degree normalization (|Nu| and |Ni|)
            deg_u = torch.bincount(src, minlength=u_emb.size(0)).float().clamp(min=1)
            deg_i = torch.bincount(dst, minlength=i_emb.size(0)).float().clamp(min=1)
            norm = 1.0 / torch.sqrt(deg_u[src] * deg_i[dst])  # [num_edges]

            u_src = u_emb[src]  # [num_edges, dim]
            i_dst = i_emb[dst]  # [num_edges, dim]

            # Message from item to user
            interaction = i_dst * u_src  # [num_edges, dim]
            msg_ui = self.linear_layers_W1[layer](i_dst) + self.linear_layers_W2[layer](interaction)
            msg_ui = self.dropout(msg_ui * norm.view(-1, 1))

            # Aggregate messages to users
            agg_user = torch.zeros_like(u_emb).index_add(0, src, msg_ui)

            # Self-connection
            self_user = self.linear_layers_W1[layer](u_emb)
            u_emb_new = self.leaky_relu(self_user + agg_user)

            # Message from user to item
            interaction2 = u_src * i_dst
            msg_iu = self.linear_layers_W1[layer](u_src) + self.linear_layers_W2[layer](interaction2)
            msg_iu = self.dropout(msg_iu * norm.view(-1, 1))

            # Aggregate messages to items
            agg_item = torch.zeros_like(i_emb).index_add(0, dst, msg_iu)

            # Self-connection
            self_item = self.linear_layers_W1[layer](i_emb)
            i_emb_new = self.leaky_relu(self_item + agg_item)

            u_emb = F.normalize(u_emb_new, dim=1)
            i_emb = F.normalize(i_emb_new, dim=1)

            all_user_embs.append(u_emb)
            all_item_embs.append(i_emb)

        final_user = torch.stack(all_user_embs, dim=1).mean(dim=1)
        final_item = torch.stack(all_item_embs, dim=1).mean(dim=1)

        return final_user, final_item
    
    def l2_regularization(self):
        return 0.5 * (self.user_embedding.weight.norm(2).pow(2) +
                    self.item_embedding.weight.norm(2).pow(2))


class GCFEncoder(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index):
        src, dst = edge_index
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        all_user_embs = [u_emb]
        all_item_embs = [i_emb]

        for _ in range(self.num_layers):
            u_src = u_emb[src]
            i_dst = i_emb[dst]
            msg_ui = u_src * i_dst
            msg_iu = i_dst * u_src

            agg_user = torch.zeros_like(u_emb).index_add(0, src, msg_iu)
            agg_item = torch.zeros_like(i_emb).index_add(0, dst, msg_ui)

            u_emb = F.normalize(u_emb + agg_user, dim=1)
            i_emb = F.normalize(i_emb + agg_item, dim=1)

            all_user_embs.append(u_emb)
            all_item_embs.append(i_emb)

        final_user = torch.stack(all_user_embs, dim=1).mean(dim=1)
        final_item = torch.stack(all_item_embs, dim=1).mean(dim=1)
        return final_user, final_item
    
    def l2_regularization(self):
        return 0.5 * (self.user_embedding.weight.norm(2).pow(2) +
                    self.item_embedding.weight.norm(2).pow(2))


class EdgeDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_user, z_business, edge_index):
        src, dst = edge_index
        h_user = z_user[src]
        h_business = z_business[dst]
        h = torch.cat([h_user, h_business], dim=1)
        return self.mlp(h).view(-1)


def load_embeddings(hdf5_path, group_name):
    with h5py.File(hdf5_path, 'r') as f:
        group = f[group_name]
        ids = list(group.keys())
    return ids


def build_index_mapping(ids):
    return {k: i for i, k in enumerate(ids)}


def stream_edges(reviews_path, user_map, biz_map):
    edges = []
    skipped = 0
    with open(reviews_path, 'r') as f:
        for line in f:
            r = json.loads(line)
            u = r['user_id']
            b = r['business_id']
            if u in user_map and b in biz_map:
                edges.append((user_map[u], biz_map[b]))
            else:
                skipped += 1
    if skipped > 0:
        print(f"⚠️ Пропущено {skipped} рёбер с неизвестными узлами (cold start фильтрация)")
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


@torch.no_grad()
def evaluate(encoder, decoder, edge_index, device):
    encoder.eval()
    decoder.eval()
    user_emb, item_emb = encoder(edge_index.to(device))
    src, dst = edge_index
    pos_logits = decoder(user_emb, item_emb, edge_index.to(device))
    pos_labels = torch.ones_like(pos_logits)

    neg_dst = torch.randint(0, item_emb.size(0), (src.size(0),), device=device)
    neg_logits = decoder(user_emb, item_emb, (src.to(device), neg_dst))
    neg_labels = torch.zeros_like(neg_logits)

    logits = torch.cat([pos_logits, neg_logits]).sigmoid().cpu().numpy()
    labels = torch.cat([pos_labels, neg_labels]).cpu().numpy()
    auc = roc_auc_score(labels, logits)
    ap = average_precision_score(labels, logits)
    return auc, ap


def train(encoder, decoder, edge_index, optimizer, criterion, device, is_eval=False, lambda_reg=1e-5):
    if not is_eval:
        encoder.train()
        decoder.train()
        optimizer.zero_grad()
    else:
        encoder.eval()
        decoder.eval()

    user_emb, item_emb = encoder(edge_index.to(device))

    src, dst = edge_index
    pos_logits = decoder(user_emb, item_emb, edge_index.to(device))
    pos_labels = torch.ones_like(pos_logits)

    neg_dst = torch.randint(0, item_emb.size(0), (src.size(0),), device=device)
    neg_logits = decoder(user_emb, item_emb, (src.to(device), neg_dst))
    neg_labels = torch.zeros_like(neg_logits)

    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([pos_labels, neg_labels])

    loss = criterion(logits, labels)

    # ➕ Добавляем L2-регуляризацию
    if hasattr(encoder, 'l2_regularization'):
        loss = loss + lambda_reg * encoder.l2_regularization()

    if not is_eval:
        loss.backward()
        optimizer.step()

    return loss.item()


def main(args):
    print("📦 Загрузка ID из эмбеддингов...")
    user_ids = load_embeddings(args.embeddings_train, "user")
    biz_ids = load_embeddings(args.embeddings_train, "business")
    user_map = build_index_mapping(user_ids)
    biz_map = build_index_mapping(biz_ids)

    print("🔄 Загрузка обучающих рёбер...")
    train_edge = stream_edges(args.train_json, user_map, biz_map)

    print("🔄 Загрузка валидационных рёбер (cold start фильтрация)...")
    val_edge = stream_edges(args.val_json, user_map, biz_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = NGCFEncoder(len(user_map), len(biz_map), args.out_dim, num_layers=args.num_layers).to(device)
    decoder = EdgeDecoder(args.out_dim).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    counter = 0
    patience = 20
    min_delta = 1e-8

    for epoch in range(1, args.epochs + 1):
        loss = train(encoder, decoder, train_edge, optimizer, criterion, device)
        val_loss = train(encoder, decoder, val_edge, optimizer=None, criterion=criterion, device=device, is_eval=True)
        print(f"[Epoch {epoch:03d}] Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
            print(f"📈 Улучшение: val_loss = {val_loss:.6f}")
        else:
            counter += 1
            print(f"⏸ Без улучшения ({counter}/{patience})")
            if counter >= patience:
                print(f"🛑 Early stopping: нет улучшения за {patience} эпох")
                break

    print("📊 Финальная оценка на валидации:")
    auc, ap = evaluate(encoder, decoder, val_edge, device)
    print(f"✅ Test ROC-AUC: {auc:.4f} | Average Precision: {ap:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_train", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    args = parser.parse_args()
    main(args)
