import time
import pickle
import queue
import threading
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn

# --- HARDWARE & TRAINING CONFIGURATION ---
DATA_DIR = "processed_data"
MODEL_OUT = "best_mercari_model.pt"
BATCH_SIZE = 2048        # ~490 MB per dense batch; ideal for 6GB VRAM + Tensor Cores
VAL_BATCH_SIZE = 4096
MAX_EPOCHS = 200
PATIENCE = 7             # Terminate if no Val RMSLE improvement for 7 epochs
INIT_LR = 2e-3
WEIGHT_DECAY = 1e-4

assert torch.cuda.is_available(), "CUDA GPU not detected! Check your PyTorch CUDA installation."
DEVICE = torch.device("cuda")

# Enable TF32 & cuDNN auto-tuner for RTX 40-series (Ada Lovelace)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
USE_BF16 = torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

torch.manual_seed(42)
np.random.seed(42)

print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Precision: {'bfloat16' if USE_BF16 else 'float16'} AMP + TF32 | Batch Size: {BATCH_SIZE} | Early Stop Patience: {PATIENCE}\n")

# --- LOAD DATA ---
print("Loading preprocessed arrays...")
X_train_tfidf = sparse.load_npz(f"{DATA_DIR}/X_train_tfidf.npz").tocsr()
X_val_tfidf = sparse.load_npz(f"{DATA_DIR}/X_val_tfidf.npz").tocsr()

tab = np.load(f"{DATA_DIR}/tabular_data.npz")
X_train_cat, X_val_cat = tab["X_train_cat"], tab["X_val_cat"]
X_train_num, X_val_num = tab["X_train_num"], tab["X_val_num"]
y_train, y_val = tab["y_train"], tab["y_val"]

with open(f"{DATA_DIR}/metadata.pkl", "rb") as f:
    vocab_sizes = pickle.load(f)["vocab_sizes"]


# --- MULTIMODAL NEURAL NETWORK ---
class MercariMultimodalNet(nn.Module):
    def __init__(self, tfidf_dim: int, vocab_sizes: dict, num_dim: int):
        super().__init__()
        cat_order = [
            "brand_clean", "category_name", "cat_main",
            "cat_sub1", "cat_sub2", "item_condition_id", "shipping"
        ]
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for col in cat_order:
            num_classes = vocab_sizes[col]
            emb_dim = min(32, max(4, (num_classes + 1) // 2))
            self.embeddings.append(nn.Embedding(num_classes, emb_dim))
            total_emb_dim += emb_dim
        self.emb_drop = nn.Dropout(0.15)

        self.text_net = nn.Sequential(
            nn.Linear(tfidf_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.35),
        )

        fusion_dim = 256 + total_emb_dim + num_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_tfidf, x_cat, x_num):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_feat = self.emb_drop(torch.cat(emb_outs, dim=1))
        text_feat = self.text_net(x_tfidf)
        fused = torch.cat([text_feat, cat_feat, x_num], dim=1)
        return self.head(fused).squeeze(1)


# --- PREFETCHED SPARSE BATCH GENERATOR ---
def prefetched_batches(X_tfidf, X_cat, X_num, y, batch_size, shuffle=True, prefetch=2):
    """Slices CSR sparse rows on a background thread with pinned host memory so GPU never waits on CPU."""
    n_samples = len(y)
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    q = queue.Queue(maxsize=prefetch)
    sentinel = object()

    def producer():
        for start in range(0, n_samples, batch_size):
            idx = indices[start : start + batch_size]
            batch = (
                torch.from_numpy(X_tfidf[idx].toarray()).pin_memory(),
                torch.from_numpy(X_cat[idx]).pin_memory(),
                torch.from_numpy(X_num[idx]).pin_memory(),
                torch.from_numpy(y[idx]).pin_memory(),
            )
            q.put(batch)
        q.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item


model = MercariMultimodalNet(
    tfidf_dim=X_train_tfidf.shape[1],
    vocab_sizes=vocab_sizes,
    num_dim=X_train_num.shape[1],
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {total_params:,}\n")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
)
scaler = torch.amp.GradScaler("cuda", enabled=(not USE_BF16))

best_rmsle = float("inf")
best_epoch = 0
epochs_no_improve = 0

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    model.train()
    train_loss_sum = 0.0

    for x_tf, x_c, x_n, target in prefetched_batches(
        X_train_tfidf, X_train_cat, X_train_num, y_train, BATCH_SIZE, shuffle=True
    ):
        x_tf = x_tf.to(DEVICE, non_blocking=True)
        x_c = x_c.to(DEVICE, non_blocking=True)
        x_n = x_n.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            preds = model(x_tf, x_c, x_n)
            loss = criterion(preds, target)

        if USE_BF16:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss_sum += loss.item() * len(target)

    train_rmsle = np.sqrt(train_loss_sum / len(y_train))

    # --- VALIDATION ---
    model.eval()
    val_preds = []
    with torch.no_grad():
        for x_tf, x_c, x_n, _ in prefetched_batches(
            X_val_tfidf, X_val_cat, X_val_num, y_val, VAL_BATCH_SIZE, shuffle=False
        ):
            x_tf = x_tf.to(DEVICE, non_blocking=True)
            x_c = x_c.to(DEVICE, non_blocking=True)
            x_n = x_n.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                preds = model(x_tf, x_c, x_n)
            val_preds.append(preds.float().cpu().numpy())

    val_preds = np.clip(np.concatenate(val_preds), 0.0, None)
    val_rmsle = np.sqrt(np.mean((val_preds - y_val) ** 2))
    scheduler.step(val_rmsle)

    elapsed = time.time() - t0
    current_lr = optimizer.param_groups[0]["lr"]
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    if val_rmsle < best_rmsle - 1e-5:
        best_rmsle = val_rmsle
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_OUT)
        status = f"  <-- BEST SAVED ({MODEL_OUT})"
    else:
        epochs_no_improve += 1
        status = f"  (no improve: {epochs_no_improve}/{PATIENCE})"

    print(
        f"Epoch {epoch:03d}/{MAX_EPOCHS} [{elapsed:4.1f}s | VRAM {peak_vram:.2f}GB | LR {current_lr:.1e}] | "
        f"Train RMSLE: {train_rmsle:.4f} | Val RMSLE: {val_rmsle:.4f}{status}"
    )

    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping triggered after {PATIENCE} epochs without improvement.")
        break

# --- EVALUATE BEST CHECKPOINT ---
print(f"\nTraining complete. Best Validation RMSLE: {best_rmsle:.4f} (at Epoch {best_epoch})")
model.load_state_dict(torch.load(MODEL_OUT, weights_only=True))
model.eval()

best_val_preds = []
with torch.no_grad():
    for x_tf, x_c, x_n, _ in prefetched_batches(
        X_val_tfidf, X_val_cat, X_val_num, y_val, VAL_BATCH_SIZE, shuffle=False
    ):
        x_tf = x_tf.to(DEVICE, non_blocking=True)
        x_c = x_c.to(DEVICE, non_blocking=True)
        x_n = x_n.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            best_val_preds.append(model(x_tf, x_c, x_n).float().cpu().numpy())

best_val_preds = np.clip(np.concatenate(best_val_preds), 0.0, None)
actual_dollars = np.expm1(y_val[:12])
pred_dollars = np.expm1(best_val_preds[:12])

print("\nSample Predictions from Best Saved Model (Actual vs. Predicted):")
for i in range(12):
    print(f"  Item {i+1:2d}: Actual = ${actual_dollars[i]:7.2f} | Predicted = ${pred_dollars[i]:7.2f}")