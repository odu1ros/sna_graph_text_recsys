import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.transforms import ToUndirected
import h5py
import json
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# torch.cuda.set_per_process_memory_fraction(0.8, 0) 

DEVICE = 'cuda'

class GNNEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        conv1_dict = {
            ('user', 'interacts', 'business'): SAGEConv((-1, -1), hidden_dim),
            ('business', 'rev_interacts', 'user'): SAGEConv((-1, -1), hidden_dim),
            ('user', 'friends_with', 'user'): SAGEConv(-1, hidden_dim),
            ('business', 'similar_to', 'business'): SAGEConv(-1, hidden_dim),
            ('business', 'belongs_to', 'category'): SAGEConv((-1,-1), hidden_dim),
            ('category', 'category_of', 'business'): SAGEConv((-1,-1), hidden_dim),
        }
        conv_out_dict = {
                ('user', 'interacts', 'business'): SAGEConv(hidden_dim, out_dim),
                ('business', 'rev_interacts', 'user'): SAGEConv(hidden_dim, out_dim),
                ('user', 'friends_with', 'user'): SAGEConv(hidden_dim, out_dim),
                ('business', 'similar_to', 'business'): SAGEConv(hidden_dim, out_dim),
                ('business', 'belongs_to', 'category'): SAGEConv((hidden_dim, hidden_dim), out_dim),
                ('category', 'category_of', 'business'): SAGEConv((hidden_dim, hidden_dim), out_dim),
            }

        self.convs = nn.ModuleList([
            HeteroConv(conv1_dict, aggr='sum'),
            HeteroConv(conv_out_dict, aggr='sum')
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


def stream_edges(reviews_path, user_map, biz_map):
    edges = []
    skipped = 0
    with open(reviews_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                u = r['user_id']
                b = r['business_id']
                if u in user_map and b in biz_map:
                    edges.append((user_map[u], biz_map[b]))
                else:
                    skipped += 1
            except json.JSONDecodeError as e:
                print(f'Ошибка декодирования JSON в строке: {e}')
                print(f'Проблемная строка: {line.strip()[:100]}...')
                skipped += 1
                continue
    if skipped > 0:
        print(f"⚠️ Пропущено {skipped} рёбер с неизвестными узлами (cold start фильтрация)")
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


@torch.no_grad()
def evaluate(encoder, decoder, loader, device=DEVICE):
    encoder.eval()
    decoder.eval()
    all_logits = []
    all_labels = []

    for batch in tqdm(loader, desc='eval batches'):
        batch = batch.to(device)
        z_dict = encoder(batch.x_dict, batch.edge_index_dict)
        
        edge_label_index = batch['user', 'interacts', 'business'].edge_label_index
        edge_label = batch['user', 'interacts', 'business'].edge_label
        
        src, dst = edge_label_index
        logits = decoder(z_dict['user'], z_dict['business'], (src, dst))
        
        all_logits.append(logits.cpu())
        all_labels.append(edge_label.cpu())

    final_logits = torch.cat(all_logits).numpy()
    final_labels = torch.cat(all_labels).numpy()
    
    auc = roc_auc_score(final_labels, final_logits)
    ap = average_precision_score(final_labels, final_logits)
    return auc, ap


def train(encoder, decoder, loader, optimizer, criterion, device='cuda'):
    is_eval = optimizer is None

    if not is_eval:
        encoder.train()
        decoder.train()
        optimizer.zero_grad()
    else:
        encoder.eval()
        decoder.eval()

    total_loss = 0.0

    for batch in tqdm(loader, desc='trainigs batches'):
        batch = batch.to(device)

        if not is_eval:
            optimizer.zero_grad()
        
        z_dict = encoder(batch.x_dict, batch.edge_index_dict)
        
        edge_label_index = batch['user', 'interacts', 'business'].edge_label_index
        edge_label = batch['user', 'interacts', 'business'].edge_label

        src, dst = edge_label_index
        logits = decoder(z_dict['user'], z_dict['business'], (src, dst))

        loss = criterion(logits, edge_label.float())
        loss.backward()

        if not is_eval:
            optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(loader)


def main(args):
    print(f'Загрузка графа из {args.graph_path}...')
    data = torch.load(args.graph_path, weights_only=False)

    print("\n=== ПРОВЕРКА ТИПОВ И ФОРМ ПРИЗНАКОВ УЗЛОВ ПОСЛЕ ЗАГРУЗКИ ===")
    all_node_types_in_data = data.node_types
    print(f'  Обнаруженные типы узлов в графе: {all_node_types_in_data}')

    for node_type in all_node_types_in_data:
        store = data[node_type]
        print(f"  Узел: '{node_type}'")
        if hasattr(store, 'x') and store.x is not None:
            print(f"    Тип data.x: {type(store.x)}")
            print(f"    Shape data.x: {store.x.shape}")

    # print("📦 Загрузка эмбеддингов (2021)...")
    user_map = {node_id: i for i, node_id in enumerate(data['user'].node_id_str)}
    biz_map = {node_id: i for i, node_id in enumerate(data['business'].node_id_str)}

    
    if hasattr(data['user'], 'node_id_str'): del data['user'].node_id_str
    if hasattr(data['business'], 'node_id_str'): del data['business'].node_id_str
    if hasattr(data['category'], 'node_id_str'): del data['category'].node_id_str
    
    data = ToUndirected()(data).to(DEVICE)

    print("🔄 Загрузка обучающих рёбер...")
    train_edge = data['user', 'interacts', 'business'].edge_index

    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=[args.num_neighbors_lvl1, args.num_neighbors_lvl2],
        edge_label_index=(('user', 'interacts', 'business'), train_edge),
        edge_label=torch.ones(train_edge.size(1)),
        batch_size=args.batch_size,
        shuffle=True,
        neg_sampling_ratio=args.neg_sampling_ratio, # > 0 
        num_workers=args.num_workers,
        pin_memory=True if DEVICE == 'cuda' else False,
    )

    print("🔄 Загрузка валидационных рёбер (фильтрация по cold start)...")
    val_edge = stream_edges(args.val_json, user_map, biz_map)

    val_loader = LinkNeighborLoader(
            data, 
            num_neighbors=[args.num_neighbors_lvl1, args.num_neighbors_lvl2],
            edge_label_index=(('user', 'interacts', 'business'), val_edge),
            edge_label=torch.ones(val_edge.size(1)),
            batch_size=args.batch_size,
            shuffle=False,
            neg_sampling_ratio=args.neg_sampling_ratio,
            num_workers=args.num_workers,
            pin_memory=True if DEVICE == 'cuda' else False,
        )    

    encoder = GNNEncoder(args.hidden_dim, args.out_dim).to(DEVICE)
    decoder = EdgeDecoder(args.out_dim).to(DEVICE)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    counter = 0
    patience = 20
    min_delta = 1e-8

    for epoch in tqdm(range(1, args.epochs + 1), desc='epoch'):
        loss = train(encoder, decoder, train_loader, optimizer, criterion)
        val_loss = train(encoder, decoder, train_loader, optimizer=None, criterion=criterion)
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
    auc, ap = evaluate(encoder, decoder, val_loader)
    print(f"✅ Test ROC-AUC: {auc:.4f} | Average Precision: {ap:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--neg_sampling_ratio", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_neighbors_lvl1", type=int, default=20)
    parser.add_argument("--num_neighbors_lvl2", type=int, default=10)
    args = parser.parse_args()
    main(args)
