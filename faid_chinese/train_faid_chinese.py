"""
faid_chinese/train_faid_chinese.py
训练入口：纯 PyTorch + argparse + 手动 epoch 循环，仿
erlangshen_SCL/train_erlangshen_scl.py 与 fusion/train_supcon_late_fusion.py
"""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import CONFIG, get_config_for_gpu
from data_loader import FaidChineseDataset, collate_fn_factory
from losses import aux_ce_loss, multi_level_loss
from model import FaidChineseModel


# ----------------------------------------------------------------------
#                              工具函数
# ----------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(labels, preds):
    macro_f1 = f1_score(labels, preds, average="macro")
    acc      = accuracy_score(labels, preds)
    per      = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "f1_hwt":   float(per[0]),
        "f1_lgt":   float(per[1]),
        "f1_hlt":   float(per[2]),
    }


def evaluate(model, loader, device, cfg):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn      = batch["attention_mask"].to(device)
            _, logits, _ = model(input_ids, attn, return_projection=False)
            all_logits.append(logits["main"].cpu())
            all_labels.append(batch["label"])
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    preds  = logits.argmax(dim=-1).numpy()
    return compute_metrics(labels, preds), preds, labels, logits


# ----------------------------------------------------------------------
#                              训练
# ----------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    cfg = get_config_for_gpu(args.gpu_mem)
    cfg.update({k: v for k, v in vars(args).items()
                if k in CONFIG and v is not None})
    cfg["batch_size"]      = args.batch_size
    cfg["num_epochs"]      = args.epochs
    cfg["learning_rate"]   = args.lr
    cfg["output_dir"]      = args.output_dir
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, model: {cfg['model_name']}")
    print(f"batch_size={cfg['batch_size']}, epochs={cfg['num_epochs']}, lr={cfg['learning_rate']}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)

    # ----------------- 数据 -----------------
    train_ds = FaidChineseDataset(cfg["train_data_path"], tokenizer, cfg)
    val_ds   = FaidChineseDataset(cfg["val_data_path"],   tokenizer, cfg)
    collate  = collate_fn_factory(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["dataloader_num_workers"], pin_memory=True,
        collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["eval_batch_size"], shuffle=False,
        num_workers=cfg["dataloader_num_workers"], pin_memory=True,
        collate_fn=collate,
    )

    # OOD val 集（可选）
    ood_loaders = {}
    for name, key in [("model_ood", "ood_model_path"),
                      ("domain_ood", "ood_domain_path"),
                      ("transform_ood", "ood_transform_path")]:
        if cfg.get(key) and Path(cfg[key]).exists():
            ds = FaidChineseDataset(cfg[key], tokenizer, cfg)
            ood_loaders[name] = DataLoader(
                ds, batch_size=cfg["eval_batch_size"], shuffle=False,
                num_workers=cfg["dataloader_num_workers"], pin_memory=True,
                collate_fn=collate,
            )

    # ----------------- 模型 / 优化器 -----------------
    model = FaidChineseModel(cfg).to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(params, lr=cfg["learning_rate"])

    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["fp16"] and torch.cuda.is_available())

    # ----------------- 训练循环 -----------------
    best_avg_f1 = -1.0
    best_state  = None
    patience    = 0
    log_path    = Path(cfg["output_dir"]) / "train_log.jsonl"
    log_f       = open(log_path, "a", encoding="utf-8")

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        t0 = time.time()
        running = defaultdict(float)
        n_step  = 0

        for step, batch in enumerate(train_loader):
            input_ids  = batch["input_ids"].to(device, non_blocking=True)
            attn       = batch["attention_mask"].to(device, non_blocking=True)
            label      = batch["label"].to(device, non_blocking=True)
            aux_label  = batch["aux_label"].to(device, non_blocking=True)
            is_mixed   = batch["is_mixed"].to(device, non_blocking=True)
            model_id   = batch["model_id"].to(device, non_blocking=True)
            domain_id  = batch["domain_id"].to(device, non_blocking=True)
            transform_id = batch["transform_id"].to(device, non_blocking=True)

            mb = {
                "label": label, "aux_label": aux_label, "is_mixed": is_mixed,
                "model_id": model_id, "domain_id": domain_id, "transform_id": transform_id,
            }

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["fp16"] and torch.cuda.is_available()):
                _, logits, z = model(input_ids, attn, return_projection=True)
                L_main = F.cross_entropy(logits["main"], label)
                L_aux  = aux_ce_loss(logits, mb)
                L_mcl  = multi_level_loss(
                    z, mb,
                    temperature=cfg["temperature"],
                    use_5level=cfg["use_5level_mcl"],
                )
                loss = (cfg["lambda_main_ce"] * L_main
                        + cfg["lambda_aux_ce"]  * L_aux
                        + cfg["lambda_mcl"]     * L_mcl)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            running["L_main"] += float(L_main.detach())
            running["L_aux"]  += float(L_aux.detach())
            running["L_mcl"]  += float(L_mcl.detach())
            running["L"]      += float(loss.detach())
            n_step += 1

            if (step + 1) % cfg["logging_steps"] == 0:
                lr_now = scheduler.get_last_lr()[0]
                msg = (f"ep{epoch} step{step+1}/{len(train_loader)} "
                       f"L={running['L']/n_step:.4f} "
                       f"Lm={running['L_main']/n_step:.4f} "
                       f"La={running['L_aux']/n_step:.4f} "
                       f"Lmcl={running['L_mcl']/n_step:.4f} "
                       f"lr={lr_now:.2e}")
                print(msg, flush=True)

        # ----------------- 验证 -----------------
        val_metrics, _, _, _ = evaluate(model, val_loader, device, cfg)
        avg_f1 = val_metrics["macro_f1"]
        msg = {"epoch": epoch, "phase": "val_in",
               **{f"val_{k}": v for k, v in val_metrics.items()},
               "time_sec": round(time.time() - t0, 1)}
        print(json.dumps(msg, ensure_ascii=False), flush=True)
        log_f.write(json.dumps(msg, ensure_ascii=False) + "\n"); log_f.flush()

        # OOD 评估
        ood_avg = []
        for name, loader in ood_loaders.items():
            m, _, _, _ = evaluate(model, loader, device, cfg)
            ood_avg.append(m["macro_f1"])
            msg = {"epoch": epoch, "phase": name, **{f"{name}_{k}": v for k, v in m.items()}}
            print(json.dumps(msg, ensure_ascii=False), flush=True)
            log_f.write(json.dumps(msg, ensure_ascii=False) + "\n"); log_f.flush()

        # 综合 score：in-dist val + OOD 宏平均
        all_f1s = [avg_f1] + ood_avg
        score = float(np.mean(all_f1s))
        print(f"==> epoch {epoch}: avg(score)={score:.4f}, in-dist F1={avg_f1:.4f}", flush=True)

        if score > best_avg_f1:
            best_avg_f1 = score
            best_state  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            torch.save({
                "model_state": best_state,
                "cfg": cfg,
                "epoch": epoch,
                "score": score,
                "val_metrics": val_metrics,
            }, Path(cfg["output_dir"]) / "best_model.pt")
            print(f"  ↳ new best, saved.", flush=True)
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                print(f"Early stop at epoch {epoch} (patience={patience})", flush=True)
                break

    log_f.close()
    # 最后再存一份 last
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "cfg": cfg, "epoch": epoch,
    }, Path(cfg["output_dir"]) / "last_model.pt")
    print(f"Done. best avg F1 = {best_avg_f1:.4f}")


# ----------------------------------------------------------------------
#                              入口
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu_mem", default="24GB")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--epochs",  type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--output_dir", default="faid_chinese/models")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
