"""
faid_chinese/build_vector_db.py
把 train+val 的 mean-pooled embedding 灌入 FAISS 索引。
- 有 faiss → IndexFlatIP（内积 = L2-norm 后的余弦相似度）
- 没装 faiss → 降级为 numpy 内存版，infer_faid_chinese.py 同样会检测
"""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import CONFIG, get_config_for_gpu
from data_loader import FaidChineseDataset, collate_fn_factory
from model import FaidChineseModel


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-6)
    return x / n


def build_index(emb: np.ndarray, use_faiss: bool):
    """返回 (backend, index_object)。backend: 'faiss' | 'numpy'。"""
    emb = _l2_normalize(emb.astype("float32"))
    if use_faiss:
        try:
            import faiss
            idx = faiss.IndexFlatIP(emb.shape[1])
            idx.add(emb)
            return "faiss", idx
        except ImportError:
            print("[build_vector_db] faiss not installed, fallback to numpy")
    return "numpy", emb           # numpy: 直接存 normalized matrix


def main():
    cfg = get_config_for_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["vector_db_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载模型 ----
    print(f"Loading model {cfg['model_name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    ckpt = torch.load(Path(cfg["output_dir"]) / "best_model.pt",
                      map_location=device, weights_only=False)
    model_cfg = ckpt["cfg"]
    model = FaidChineseModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- 收集 train+val embedding ----
    collate = collate_fn_factory(tokenizer.pad_token_id)
    sources = [(cfg["train_data_path"], "train"), (cfg["val_data_path"], "val")]
    all_emb, all_lab, all_aux, all_mix, all_mid, all_did, all_tid, all_ids = (
        [], [], [], [], [], [], [], []
    )
    for path, tag in sources:
        print(f"Encoding {tag}: {path}")
        ds = FaidChineseDataset(path, tokenizer, model_cfg)
        loader = DataLoader(ds, batch_size=cfg["eval_batch_size"],
                            shuffle=False, num_workers=2, collate_fn=collate)
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                pooled, _, _ = model(input_ids, attn, return_projection=False)
                pooled = pooled.cpu().numpy()
                all_emb.append(pooled)
                all_lab.extend(batch["label"].tolist())
                all_aux.extend(batch["aux_label"].tolist())
                all_mix.extend(batch["is_mixed"].tolist())
                all_mid.extend(batch["model_id"].tolist())
                all_did.extend(batch["domain_id"].tolist())
                all_tid.extend(batch["transform_id"].tolist())
                all_ids.extend(batch["item_id"])

    emb = np.concatenate(all_emb, axis=0).astype("float32")
    print(f"Total embeddings: {emb.shape}")

    # ---- 建库 ----
    backend, idx = build_index(emb, cfg.get("use_faiss", True))

    # ---- 持久化 ----
    np.save(out_dir / f"{cfg['vector_db_name']}_emb.npy", emb)
    if backend == "faiss":
        import faiss
        faiss.write_index(idx, str(out_dir / f"{cfg['vector_db_name']}.index"))
    else:
        # 写一个 marker 文件，告诉 inference 走 numpy 路径
        (out_dir / f"{cfg['vector_db_name']}_USE_NUMPY.flag").touch()
    meta = {
        "backend": backend,
        "num":     int(emb.shape[0]),
        "dim":     int(emb.shape[1]),
        "ids":     all_ids,
        "label":   all_lab,
        "aux_label":all_aux,
        "is_mixed":all_mix,
        "model_id":all_mid,
        "domain_id":all_did,
        "transform_id":all_tid,
    }
    with open(out_dir / f"{cfg['vector_db_name']}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"Saved vector DB (backend={backend}) to {out_dir}")
    print(f"  - {cfg['vector_db_name']}_emb.npy      ({emb.nbytes/1024/1024:.1f} MB)")
    if backend == "faiss":
        print(f"  - {cfg['vector_db_name']}.index")
    else:
        print(f"  - {cfg['vector_db_name']}_USE_NUMPY.flag")
    print(f"  - {cfg['vector_db_name']}_meta.json")


if __name__ == "__main__":
    main()
