import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.transforms import ToUndirected
import h5py
import json
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score


class GNNEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.convs = nn.ModuleList([
            HeteroConv({
                ('user', 'interacts', 'business'): SAGEConv((-1, -1), hidden_dim),
                ('business', 'rev_interacts', 'user'): SAGEConv((-1, -1), hidden_dim),
            }, aggr='sum'),
            HeteroConv({
                ('user', 'interacts', 'business'): SAGEConv((-1, -1), out_dim),
                ('business', 'rev_interacts', 'user'): SAGEConv((-1, -1), out_dim),
            }, aggr='sum')
        ])

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return x_dict


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
        data = [group[k][()] for k in ids]
    return ids, torch.tensor(np.stack(data), dtype=torch.float)


def build_index_mapping(ids):
    return {k: i for i, k in enumerate(ids)}


def stream_edges(reviews_path, user_map, biz_map, allow_unknown=False):
    edges = []
    skipped = 0
    with open(reviews_path, 'r') as f:
        for line in f:
            r = json.loads(line)
            u = r['user_id']
            b = r['business_id']
            if u in user_map and b in biz_map:
                edges.append((user_map[u], biz_map[b]))
            elif not allow_unknown:
                skipped += 1
    if not allow_unknown and skipped > 0:
        print(f"⚠️  Пропущено {skipped} рёбер с неизвестными узлами из {reviews_path}")
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


@torch.no_grad()
def evaluate(model, decoder, data, edge_index):
    model.eval()
    decoder.eval()
    z_dict = model(data.x_dict, data.edge_index_dict)
    pos_src, pos_dst = edge_index
    pos_logits = decoder(z_dict['user'], z_dict['business'], edge_index)
    pos_labels = torch.ones_like(pos_logits)

    neg_dst = torch.randint(0, z_dict['business'].size(0), (pos_src.size(0),), device=pos_src.device)
    neg_logits = decoder(z_dict['user'], z_dict['business'], (pos_src, neg_dst))
    neg_labels = torch.zeros_like(neg_logits)

    logits = torch.cat([pos_logits, neg_logits]).sigmoid().cpu().numpy()
    labels = torch.cat([pos_labels, neg_labels]).cpu().numpy()

    auc = roc_auc_score(labels, logits)
    ap = average_precision_score(labels, logits)
    return auc, ap


def train(model, decoder, data, edge_index, optimizer, criterion):
    is_eval = optimizer is None

    if not is_eval:
        model.train()
        decoder.train()
        optimizer.zero_grad()
    else:
        model.eval()
        decoder.eval()

    with torch.set_grad_enabled(not is_eval):
        z_dict = model(data.x_dict, data.edge_index_dict)

        pos_src, pos_dst = edge_index
        pos_logits = decoder(z_dict['user'], z_dict['business'], edge_index)
        pos_labels = torch.ones_like(pos_logits)

        neg_dst = torch.randint(0, z_dict['business'].size(0), (pos_src.size(0),), device=pos_src.device)
        neg_logits = decoder(z_dict['user'], z_dict['business'], (pos_src, neg_dst))
        neg_labels = torch.zeros_like(neg_logits)

        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([pos_labels, neg_labels])

        loss = criterion(logits, labels)

        if not is_eval:
            loss.backward()
            optimizer.step()

        return loss.item()


def main(args):
    print("📦 Загрузка train-эмбеддингов...")
    train_user_ids, train_user_x = load_embeddings(args.embeddings_train, "user")
    train_biz_ids, train_biz_x = load_embeddings(args.embeddings_train, "business")

    print("📦 Загрузка валид. эмбеддингов...")
    val_user_ids, val_user_x = load_embeddings(args.embeddings_val, "user")
    val_biz_ids, val_biz_x = load_embeddings(args.embeddings_val, "business")

    # Только общие узлы между train/val
    shared_users = sorted(set(train_user_ids) & set(val_user_ids))
    shared_businesses = sorted(set(train_biz_ids) & set(val_biz_ids))

    print(f"👥 Пользователей валидации, пересекающихся с train: {len(shared_users)}")
    print(f"🏢 Бизнесов валидации, пересекающихся с train: {len(shared_businesses)}")

    # Строим общую матрицу узлов по train
    user_map = build_index_mapping(train_user_ids)
    biz_map = build_index_mapping(train_biz_ids)

    print("🔄 Загрузка обучающих рёбер...")
    train_edge = stream_edges(args.train_json, user_map, biz_map, allow_unknown=False)

    print("🔄 Загрузка валидационных рёбер...")
    val_edge = stream_edges(args.val_json, user_map, biz_map, allow_unknown=False)

    print("🧠 Построение графа...")
    data = HeteroData()
    data['user'].x = train_user_x
    data['business'].x = train_biz_x
    data['user', 'interacts', 'business'].edge_index = train_edge
    data['business', 'rev_interacts', 'user'].edge_index = train_edge[[1, 0]]
    data = ToUndirected()(data).to('cuda')

    encoder = GNNEncoder(args.hidden_dim, args.out_dim).to('cuda')
    decoder = EdgeDecoder(args.out_dim).to('cuda')

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience = 20
    counter = 0
    min_delta = 0.0001

    for epoch in range(1, args.epochs + 1):
        loss = train(encoder, decoder, data, train_edge.to('cuda'), optimizer, criterion)
        val_loss = train(encoder, decoder, data, val_edge.to('cuda'), optimizer=None, criterion=criterion)
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

    print("📊 Финальная оценка на 2022")
    auc, ap = evaluate(encoder, decoder, data, val_edge.to('cuda'))
    print(f"✅ Test ROC-AUC: {auc:.4f} | Average Precision: {ap:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_train", type=str, required=True)
    parser.add_argument("--embeddings_val", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=64)
    args = parser.parse_args()
    main(args)
