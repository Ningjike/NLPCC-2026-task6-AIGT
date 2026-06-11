"""
Erlangshen SCL - EDA-augmented Supervised Contrastive Learning (Khosla2020).

view1 = original text
view2 = EDA(text): random token deletion (10%) + random [MASK] replacement (15%)
The two views are mutual positives in SCL.

Loss: L = L_CE + lambda * L_SupCon, lambda =0.05 (lightweight; SCL is auxiliary).
Uses pooler_output (CLS + Dense + Tanh) to match Erlangshen NLI pretraining.
"""

import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import json
import gc
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, DataCollatorWithPadding
from sklearn.metrics import f1_score, accuracy_score


def eda_augment(input_ids, attention_mask, tokenizer, p_del=0.10, p_mask=0.15):
    """Token-level EDA: random deletion + random [MASK] replacement.

    Operates on tokenized batches (after padding). Each non-special, non-pad
    token has independent probability of being deleted or masked.
    """
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id

    B, L = input_ids.shape

    # Eligible positions: attended (non-padding) and not a special token.
    is_special = torch.zeros_like(input_ids, dtype=torch.bool)
    for tid in (pad_id, cls_id, sep_id):
        if tid is not None:
            is_special |= (input_ids == tid)
    eligible = (attention_mask ==1) & ~is_special

    # Independent per-token coin flips. 
    rand_del = torch.rand(B, L, device=input_ids.device) < p_del
    rand_mask = torch.rand(B, L, device=input_ids.device) < p_mask
    del_pos = rand_del & eligible
    # Do not mask a token we are about to delete.
    mask_pos = rand_mask & eligible & ~del_pos

    aug_input_ids = input_ids.clone()
    if mask_id is not None:
        aug_input_ids = torch.where(mask_pos, mask_id, aug_input_ids)
    aug_input_ids = torch.where(del_pos, pad_id, aug_input_ids)
    aug_attention_mask = attention_mask.clone() & ~del_pos

    return aug_input_ids, aug_attention_mask


class TextClassificationDataset(Dataset):
	def __init__(self, data_path, tokenizer, max_length=256):
		with open(data_path, 'r', encoding='utf-8') as f:
			self.data = json.load(f)
		self.tokenizer = tokenizer
		self.max_length = max_length
		print(f"Loaded dataset: {len(self.data)} samples")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		item = self.data[idx]
		encoding = self.tokenizer(
			item['text'],
			max_length=self.max_length,
			truncation=True,
			padding=False,
			return_tensors=None
		)
		return {
			'input_ids': encoding['input_ids'],
			'attention_mask': encoding['attention_mask'],
			'labels': item['label'],
		}


class SupConProjectionHead(nn.Module):
	def __init__(self, input_dim: int, output_dim: int =128, dropout: float =0.1):
		super().__init__()
		self.projection = nn.Sequential(
			nn.Linear(input_dim, input_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(input_dim, output_dim),
		)

	def forward(self, x):
		return F.normalize(self.projection(x), dim=1)


class ErlangshenSCLModel(nn.Module):
	def __init__(
		self,
		model_name: str = 'IDEA-CCNL/Erlangshen-Roberta-110M-NLI',
		num_labels: int =3,
		projection_dim: int =128,
		dropout: float =0.1,
		supcon_lambda: float =0.05,
	):
		super().__init__()
		self.num_labels = num_labels
		self.model_name = model_name
		self.supcon_lambda = supcon_lambda

		self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
		h = self.encoder.config.hidden_size

		self.classifier = nn.Linear(h, num_labels)
		self.projection_head = SupConProjectionHead(h, projection_dim, dropout)

	def forward(self, input_ids, attention_mask, aug_input_ids=None, aug_attention_mask=None):
		"""Returns (logits, scl_features).

		view1 = original input (always computed).
		view2 = augmented input if provided; otherwise single-view.
		scl_features is set only when training AND view2 is provided AND
		supcon_lambda >0, so eval/inference takes a single encoder pass.
		"""
		out1 = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
		p1 = out1.pooler_output
		logits = self.classifier(p1)

		if self.training and aug_input_ids is not None and self.supcon_lambda >0:
			out2 = self.encoder(
				input_ids=aug_input_ids,
				attention_mask=aug_attention_mask,
				return_dict=True
			)
			p2 = out2.pooler_output
			scl_features = self.projection_head(torch.cat([p1, p2], dim=0))
		else:
			scl_features = None

		return logits, scl_features


def supcon_loss(features, labels, temperature: float =0.07):
	"""SupCon loss. features: (N, D) L2-normalized, labels: (N,).
	N =2B with view1+view2 concatenated (labels tiled)."""
	device = features.device
	n = features.shape[0]

	labels = labels.contiguous().view(-1,1)
	mask = torch.eq(labels, labels.T).float().to(device)

	sim = torch.div(features @ features.T, temperature)
	sim = sim - sim.max(dim=1, keepdim=True).values.detach()

	logits_mask = torch.ones_like(mask).scatter(
		1, torch.arange(n, device=device).view(-1,1),0
	)
	mask = mask * logits_mask

	exp_logits = torch.exp(sim) * logits_mask
	log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True) +1e-12)

	pos_per_anchor = mask.sum(dim=1)
	pos_per_anchor = torch.where(
		pos_per_anchor <1e-6, torch.ones_like(pos_per_anchor), pos_per_anchor
	)
	mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_per_anchor

	return -mean_log_prob_pos.mean()


def compute_metrics(labels, preds):
	macro_f1 = f1_score(labels, preds, average='macro')
	accuracy = accuracy_score(labels, preds)
	f1_per_class = f1_score(labels, preds, average=None, labels=[0,1,2], zero_division=0)
	return {
		'macro_f1': macro_f1,
		'accuracy': accuracy,
		'f1_hwt': f1_per_class[0],
		'f1_lgt': f1_per_class[1],
		'f1_hlt': f1_per_class[2],
	}


def train_epoch(model, dataloader, optimizer, scheduler, device, supcon_lambda, tokenizer,
	p_del=0.10, p_mask=0.15):
	model.train()
	total_loss = total_ce = total_scl =0
	all_preds, all_labels = [], []

	for batch_idx, batch in enumerate(dataloader):
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		labels = batch['labels'].to(device)

		# Generate EDA-augmented view2.
		aug_ids, aug_mask = eda_augment(input_ids, attention_mask, tokenizer, p_del, p_mask)

		optimizer.zero_grad()
		logits, scl_features = model(input_ids, attention_mask, aug_ids, aug_mask)

		ce_loss = F.cross_entropy(logits, labels)
		if scl_features is not None and supcon_lambda >0:
			labels_tiled = torch.cat([labels, labels])
			scl_loss = supcon_loss(scl_features, labels_tiled)
		else:
			scl_loss = torch.tensor(0.0, device=device)

		loss = ce_loss + supcon_lambda * scl_loss
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		scheduler.step()

		total_loss += loss.item()
		total_ce += ce_loss.item()
		total_scl += scl_loss.item() if torch.isfinite(scl_loss) else 0.0

		all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
		all_labels.extend(labels.cpu().numpy())

		if (batch_idx +1) %50 ==0:
			m = compute_metrics(all_labels, all_preds)
			print(f" Batch {batch_idx +1}/{len(dataloader)} | "
				f"Loss: {total_loss / (batch_idx +1):.4f} | "
				f"CE: {total_ce / (batch_idx +1):.4f} | "
				f"SCL: {total_scl / (batch_idx +1):.4f} | "
				f"Macro F1: {m[m["macro_f1"]]:.4f}")

	return total_loss / len(dataloader), compute_metrics(all_labels, all_preds)


def evaluate(model, dataloader, device):
	model.eval()
	total_loss =0
	all_preds, all_labels = [], []

	with torch.no_grad():
		for batch in dataloader:
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)

			logits, _ = model(input_ids, attention_mask)
			total_loss += F.cross_entropy(logits, labels).item()
			all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
			all_labels.extend(labels.cpu().numpy())

	return total_loss / len(dataloader), compute_metrics(all_labels, all_preds)


def train_model(
	model_name='IDEA-CCNL/Erlangshen-Roberta-330M-NLI',
	train_data_path='data/processed/train.json',
	val_data_path='data/processed/val.json',
	output_dir='erlangshen_SCL/models',
	max_length=256,
	batch_size=16,
	num_epochs=10,
	learning_rate=2e-5,
	warmup_ratio=0.1,
	weight_decay=0.01,
	projection_dim=128,
	dropout=0.1,
	supcon_lambda=0.05,
	p_del=0.10,
	p_mask=0.15,
	eval_steps=200,
	logging_steps=50,
):
	print('=' *60)
	print('=== Erlangshen SCL (EDA view augmentation) ===')
	print('=' *60)
	print(f'model={model_name} lambda={supcon_lambda} EDA(p_del={p_del}, p_mask={p_mask})')

	if not torch.cuda.is_available():
		raise RuntimeError('CUDA required')

	device = torch.device('cuda')
	print(f'GPU: {torch.cuda.get_device_name(0)}')
	print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory /1024**3:.2f} GB')

	gc.collect()
	torch.cuda.empty_cache()

	tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
	model = ErlangshenSCLModel(
		model_name, num_labels=3, projection_dim=projection_dim,
		dropout=dropout, supcon_lambda=supcon_lambda
	).to(device)
	print(f'params: {sum(p.numel() for p in model.parameters()) /1e6:.2f}M')
	print(f'VRAM after load: {torch.cuda.memory_allocated() /1024**3:.2f} GB')

	train_dataset = TextClassificationDataset(train_data_path, tokenizer, max_length)
	val_dataset = TextClassificationDataset(val_data_path, tokenizer, max_length)

	train_loader = DataLoader(
		train_dataset, batch_size=batch_size, shuffle=True,
		collate_fn=DataCollatorWithPadding(tokenizer), num_workers=0, pin_memory=True
	)
	val_loader = DataLoader(
		val_dataset, batch_size=batch_size *2, shuffle=False,
		collate_fn=DataCollatorWithPadding(tokenizer), num_workers=0, pin_memory=True
	)
	print(f'train: {len(train_dataset)} samples, {len(train_loader)} batches')
	print(f'val: {len(val_dataset)} samples, {len(val_loader)} batches')

	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	total_steps = len(train_loader) * num_epochs
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=int(total_steps * warmup_ratio),
		num_training_steps=total_steps
	)
	print(f'total_steps: {total_steps}, warmup: {int(total_steps * warmup_ratio)}')

	best_f1 =0
	best_state = None
	patience_counter =0
	patience =5

	for epoch in range(num_epochs):
		print(f'\n--- Epoch {epoch +1}/{num_epochs} ---')

		train_loss, train_metrics = train_epoch(
			model, train_loader, optimizer, scheduler, device,
			supcon_lambda, tokenizer, p_del, p_mask
		)
		print(f'train: loss={train_loss:.4f} f1={train_metrics["macro_f1"]:.4f} acc={train_metrics["accuracy"]:.4f}')

		val_loss, val_metrics = evaluate(model, val_loader, device)
		print(f'val: loss={val_loss:.4f} f1={val_metrics["macro_f1"]:.4f} acc={val_metrics["accuracy"]:.4f} | HWT={val_metrics["f1_hwt"]:.4f} LGT={val_metrics["f1_lgt"]:.4f} HLT={val_metrics["f1_hlt"]:.4f}')

		if val_metrics["macro_f1"] > best_f1:
			best_f1 = val_metrics["macro_f1"]
			best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
			patience_counter =0
			print(f'[best] saved (f1={best_f1:.4f})')
		else:
			patience_counter +=1
			print(f'no improve ({patience_counter}/{patience})')
			if patience_counter >= patience:
				print(f'\nearly stop: {patience} epochs without improvement')
				break

	if best_state is not None:
		model.load_state_dict(best_state)

	print('\n' + '=' *60)
	print(f'done | best macro F1: {best_f1:.4f}')
	print('=' *60)

	os.makedirs(output_dir, exist_ok=True)
	save_path = os.path.join(output_dir, 'best_model')
	os.makedirs(save_path, exist_ok=True)
	torch.save({
		'model_state_dict': model.state_dict(),
		'config': {
			'model_name': model_name, 'num_labels':3,
			'projection_dim': projection_dim, 'dropout': dropout,
			'supcon_lambda': supcon_lambda, 'p_del': p_del, 'p_mask': p_mask,
		},
		'best_f1': best_f1,
	}, os.path.join(save_path, 'model.pt'))
	tokenizer.save_pretrained(save_path)
	print(f'saved to: {save_path}')

	final_loss, final_metrics = evaluate(model, val_loader, device)
	print(f'\nfinal: f1={final_metrics["macro_f1"]:.4f} | HWT={final_metrics["f1_hwt"]:.4f} LGT={final_metrics["f1_lgt"]:.4f} HLT={final_metrics["f1_hlt"]:.4f}')
	return model, final_metrics


def main():
	train_model(
		model_name='IDEA-CCNL/Erlangshen-Roberta-330M-NLI',
		train_data_path='data/processed/train.json',
		val_data_path='data/processed/val.json',
		output_dir='erlangshen_SCL/models',
		max_length=256,
		batch_size=16,
		num_epochs=10,
		learning_rate=2e-5,
		warmup_ratio=0.1,
		weight_decay=0.01,
		projection_dim=128,
		dropout=0.1,
		supcon_lambda=0.05,
		p_del=0.10,
		p_mask=0.15,
		eval_steps=200,
		logging_steps=50,
	)


if __name__ == '__main__':
	main()
