"""
faid_chinese/infer_faid_chinese.py
推理：encoder 嵌入 → FAISS top-K → 软投票 + 主头分类
最终 logits = (1-α) · softmax(head_main) + α · softmax(knn_vote)
"""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import CONFIG, get_config_for_gpu
from data_loader import FaidChineseDataset, collate_fn_factory
from model import FaidChineseModel


# ----------------------------------------------------------------------
#                  向量库后端（faiss / numpy）
# ----------------------------------------------------------------------

class VectorDB:
    def __init__(self, cfg):
        out_dir = Path(cfg["vector_db_dir"])
        self.backend = "faiss"
        if (out_dir / f"{cfg['vector_db_name']}_USE_NUMPY.flag").exists():
            self.backend = "numpy"
        if self.backend == "faiss":
            import faiss
            self.idx = faiss.read_index(str(out_dir / f"{cfg['vector_db_name']}.index"))
        else:
            self.idx = np.load(out_dir / f"{cfg['vector_db_name']}_emb.npy")
        with open(out_dir / f"{cfg['vector_db_name']}_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.ids          = meta["ids"]
        self.label        = np.array(meta["label"],        dtype=np.int64)
        self.aux_label    = np.array(meta["aux_label"],    dtype=np.int64)
        self.is_mixed     = np.array(meta["is_mixed"],     dtype=np.int64)
        self.model_id     = np.array(meta["model_id"],     dtype=np.int64)
        self.domain_id    = np.array(meta["domain_id"],    dtype=np.int64)
        self.transform_id = np.array(meta["transform_id"], dtype=np.int64)
        print(f"[VectorDB] backend={self.backend}, n={len(self.ids)}, "
              f"label_dist={dict(zip(*np.unique(self.label, return_counts=True)))}")

    def search(self, q_emb: np.ndarray, k: int):
        """q_emb: (B, D) L2-normalized; 返回 (D, k) 的距离与索引。"""
        q = q_emb.astype("float32")
        if self.backend == "faiss":
            D, I = self.idx.search(q, k)
        else:
            # numpy: q (B,D) @ db (N,D)^T  → (B,N)
            scores = q @ self.idx.T
            idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
            # 局部排序使返回按相似度降序
            row = np.arange(scores.shape[0])[:, None]
            order = np.argsort(-scores[row, idx], axis=1)
            I = idx[row, order]
            D = scores[row, I]
        return D, I


def knn_vote(D: np.ndarray, I: np.ndarray, db: VectorDB, num_classes: int = 3,
             temperature: float = 0.7) -> np.ndarray:
    """
    D, I: (B, K)；返回 (B, num_classes) 软投票概率。
    权重 = softmax(D / temperature)，按主标签聚合。
    """
    weights = np.exp(D / max(temperature, 1e-6))
    weights = weights / weights.sum(axis=1, keepdims=True)        # (B, K)
    B, K = I.shape
    prob = np.zeros((B, num_classes), dtype=np.float32)
    for b in range(B):
        for j in range(K):
            prob[b, db.label[I[b, j]]] += weights[b, j]
    return prob


# ----------------------------------------------------------------------
#                              推理
# ----------------------------------------------------------------------

def predict_main_head(model, loader, device):
    """返回 (ids, head_probs)。"""
    ids_all, probs_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            pooled, logits, _ = model(input_ids, attn, return_projection=False)
            p = F.softmax(logits["main"], dim=-1).cpu().numpy()
            ids_all.extend(batch["item_id"])
            probs_all.append(p)
    return ids_all, np.concatenate(probs_all, axis=0)


def encode_only(model, loader, device):
    """返回 (ids, embeddings, pool_mean)。"""
    ids_all, emb_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            pooled, _, _ = model(input_ids, attn, return_projection=False)
            ids_all.extend(batch["item_id"])
            emb_all.append(pooled.cpu().numpy())
    return ids_all, np.concatenate(emb_all, axis=0)


def _l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-6)
    return x / n


def infer_one(model, db: VectorDB, path: str, cfg: dict, device,
              return_labels: bool = False):
    """对单个 JSON 跑推理，返回 (ids, preds, probs[, labels])。"""
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    # is_test=True：不过滤 ID 解析失败的行（如 testp1-0），容忍 label=None
    is_test = not return_labels
    ds = FaidChineseDataset(path, tokenizer, cfg, is_test=is_test)
    collate = collate_fn_factory(tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=cfg["eval_batch_size"], shuffle=False,
                        num_workers=2, collate_fn=collate)

    ids, head_probs = predict_main_head(model, loader, device)
    _, emb          = encode_only(model, loader, device)
    emb = _l2(emb.astype("float32"))

    D, I = db.search(emb, cfg["top_k"])
    knn_probs = knn_vote(D, I, db, num_classes=cfg["num_labels"],
                         temperature=cfg["knn_temperature"])

    alpha = cfg["alpha_knn"]
    final = (1 - alpha) * head_probs + alpha * knn_probs
    preds = final.argmax(axis=-1)

    if return_labels:
        return ids, preds, final, [it["label"] for it in ds.items]
    return ids, preds, final, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default=CONFIG["testp1_path"])
    p.add_argument("--output", default="faid_chinese/predictions.json")
    args = p.parse_args()

    cfg = get_config_for_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {cfg['output_dir']}/best_model.pt ...")
    ckpt = torch.load(Path(cfg["output_dir"]) / "best_model.pt",
                      map_location=device, weights_only=False)
    model = FaidChineseModel(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Loading vector DB ...")
    db = VectorDB(cfg)

    print(f"Predicting on {args.input} ...")
    ids, preds, probs, _ = infer_one(model, db, args.input, cfg, device,
                                     return_labels=False)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([{"id": i, "label": int(p_), "probs": probs[k].tolist()}
                   for k, (i, p_) in enumerate(zip(ids, preds))],
                  f, ensure_ascii=False, indent=2)
    print(f"Saved {len(ids)} predictions to {out_path}")


if __name__ == "__main__":
    main()
