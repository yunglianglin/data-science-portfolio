#########----- 1. PREPROCESSING SAMPLES ---------- #######
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

# === Start timer ===
start_time = datetime.now()

# === Prepare grouping keys ===
uniq_group = list(df_2.groupby(['CUSTID', 'TOKENID']).groups.keys())
tot_session = len(uniq_group)
print(f"Total sessions: {tot_session}")

# === Initialize lists ===
seq_feature_list = []
y_list = []
static_feature_list = []
actual_lens = []
token_list = []

# === Define continuous feature columns ===
seq_cont_cols = ['ACTION_CNT', 'DWELL_TIME']
static_cont_cols = ['TIME_PL_BUTN', 'CNT_LOAN_PAGE']

# === Fit scalers ===
scaler_seq = StandardScaler().fit(df_1[seq_cont_cols])
scaler_static = StandardScaler().fit(df_static_feature[static_cont_cols])

# === Iterate through each session (CUSTID + TOKENID) ===
for idx, (cust_id, token_id) in enumerate(uniq_group, start=1):
    group = df_2[(df_2['CUSTID'] == cust_id) & (df_2['TOKENID'] == token_id)]
    group = group.sort_values(by=['STEP_SEQ'], ascending=True)

    # Select last 20 steps per session
    feature_cols = ['PAGE_ID', 'ACTION_CNT', 'DWELL_TIME', 'YN_LOAN_RELATED']
    seq_data = group.iloc[-20:][feature_cols].copy()

    # Normalize continuous features
    seq_data[seq_cont_cols] = scaler_seq.transform(seq_data[seq_cont_cols])

    # Convert to numpy
    x = seq_data.values
    seq_feature_list.append(x)
    actual_lens.append(x.shape[0])
    token_list.append((cust_id, token_id))

    # Label for this session
    y = group.iloc[0]['YN_APLY_SESSION']
    y_list.append(y)

    # Static features
    static_prt_cols = ['PRT_LOAN_PAGE', 'PRT_PL_BUTN', 'PRT_LOAN_PAGE_DWELL']
    prt_vars = group.iloc[0][static_prt_cols].values
    cont_vars = scaler_static.transform(group.iloc[[0]][static_cont_cols])[0]
    static_feature = np.concatenate([prt_vars, cont_vars])
    static_feature_list.append(static_feature)

    # Log progress periodically (not every row for speed)
    if idx % 100 == 0 or idx == tot_session:
        print(f"Processed: {idx}/{tot_session} ({idx / tot_session:.1%})")

# === Convert to tensors ===
X_tensor_list = [torch.tensor(x, dtype=torch.float32) for x in seq_feature_list]
y_tensor = torch.tensor(y_list, dtype=torch.int64).unsqueeze(1)
static_tensor = torch.tensor(static_feature_list, dtype=torch.float32)

# === Pad sequences to equal length ===
seq_tensor_pad = pad_sequence(X_tensor_list, batch_first=True)

# === Log total time ===
end_time = datetime.now()
print(f"Start time : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"End time   : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duration   : {end_time - start_time}")


#########----- 2. SPLIT DATA INTO TRAINING, VALIDATION, AND TEST SETS ---------- #######
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch

RANDOM_STATE = 100

# First split: train + val vs. test (80/20)
X_seq_train_tmp, X_seq_test, X_static_train_tmp, X_static_test, y_train_tmp, y_test, len_train_tmp, len_test = train_test_split(
    seq_tensor_pad,
    static_tensor,
    y_tensor,
    actual_lens,
    test_size=0.2,
    random_state=RANDOM_STATE,
    shuffle=True
)

# Second split: training vs. validation (80/20 of train_tmp)
X_seq_train, X_seq_val, X_static_train, X_static_val, y_train, y_val, len_train, len_val = train_test_split(
    X_seq_train_tmp,
    X_static_train_tmp,
    y_train_tmp,
    len_train_tmp,
    test_size=0.2,
    random_state=RANDOM_STATE,
    shuffle=True
)

# Convert actual_lens to tensors (for DataLoader compatibility)
len_train_tensor = torch.tensor(len_train, dtype=torch.int64)
len_val_tensor   = torch.tensor(len_val, dtype=torch.int64)
len_test_tensor  = torch.tensor(len_test, dtype=torch.int64)

# === Wrap datasets into TensorDataset ===
train_dataset = TensorDataset(X_seq_train, X_static_train, y_train, len_train_tensor)
val_dataset   = TensorDataset(X_seq_val,   X_static_val,   y_val,   len_val_tensor)
test_dataset  = TensorDataset(X_seq_test,  X_static_test,  y_test,  len_test_tensor)

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LoanInterestLSTM(nn.Module):
    """
    LoanInterestLSTM:
    A neural architecture combining embeddings, bidirectional LSTM, attention, 
    and static features for loan conversion prediction.

    Parameters
    ----------
    num_pages : int
        Number of unique page IDs (for embedding layer).
    embedding_size : int
        Size of the page ID embedding vector.
    seq_input_size : int
        Number of continuous features per time step (e.g., dwell time, action count).
    static_input_size : int
        Number of static (non-sequential) features per session.
    hidden_size : int
        Number of hidden units in each LSTM layer.
    dense_size : int
        Number of units in the first dense layer.
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout rate for regularization.
    """

    def __init__(
        self,
        num_pages: int,
        embedding_size: int,
        seq_input_size: int,
        static_input_size: int,
        hidden_size: int,
        dense_size: int,
        lstm_layers: int,
        dropout: float
    ):
        super().__init__()

        # === Embedding Layer ===
        # Converts discrete page IDs into dense representations.
        self.page_embedding = nn.Embedding(
            num_embeddings=num_pages,
            embedding_dim=embedding_size,
            padding_idx=0  # ignore padding index during training
        )

        # === Bidirectional LSTM Layer ===
        self.lstm = nn.LSTM(
            input_size=embedding_size + seq_input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # === Attention Layer ===
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # === MLP for Static Features ===
        self.static_mlp = nn.Sequential(
            nn.Linear(static_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(dropout)
        )

        # === Dense Layers for Final Classification ===
        self.dense1 = nn.Linear(hidden_size * 2 + 16, dense_size)
        self.ln1 = nn.LayerNorm(dense_size)

        self.dense2 = nn.Linear(dense_size, 64)
        self.ln2 = nn.LayerNorm(64)

        self.dense3 = nn.Linear(64, 32)
        self.ln3 = nn.LayerNorm(32)

        self.dense4 = nn.Linear(32, 16)
        self.ln4 = nn.LayerNorm(16)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # === Output Layer ===
        self.output_layer = nn.Linear(16, 1)  # sigmoid applied in loss function if BCEWithLogitsLoss is used


    def forward(self, pageid_tensor, features_tensor, static_tensor, actual_len):
        """
        Forward pass through the model.

        Parameters
        ----------
        pageid_tensor : torch.Tensor
            Tensor of page IDs with shape (batch_size, seq_len).
        features_tensor : torch.Tensor
            Continuous sequence features with shape (batch_size, seq_len, seq_input_size).
        static_tensor : torch.Tensor
            Static features with shape (batch_size, static_input_size).
        actual_len : torch.Tensor
            Actual (unpadded) sequence lengths for each batch element.

        Returns
        -------
        torch.Tensor
            Predicted logits (before sigmoid) of shape (batch_size, 1).
        """
        # === Sequence Embedding ===
        pageid_embed = self.page_embedding(pageid_tensor)  # (batch_size, seq_len, embedding_size)
        lstm_input = torch.cat([pageid_embed, features_tensor], dim=-1)

        # === Packed LSTM ===
        packed = pack_padded_sequence(lstm_input, actual_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        lstm_out = self.lstm_dropout(lstm_out)

        # === Attention Mechanism ===
        attn_scores = self.attention(lstm_out).squeeze(-1)
        max_len = lstm_out.size(1)
        mask = torch.arange(max_len, device=actual_len.device).unsqueeze(0) >= actual_len.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        attn_applied = torch.sum(lstm_out * attn_weights, dim=1)

        # === Combine with Static Features ===
        if static_tensor is not None:
            static_out = self.static_mlp(static_tensor)
            dense_input = torch.cat([attn_applied, static_out], dim=-1)
        else:
            dense_input = attn_applied

        # === Dense Layers ===
        x = self.relu(self.ln1(self.dense1(dense_input)))
        x = self.relu(self.ln2(self.dense2(x)))
        x = self.relu(self.ln3(self.dense3(x)))
        x = self.relu(self.ln4(self.dense4(x)))

        # === Output Layer (no sigmoid; handle in loss) ===
        logits = self.output_layer(x)
        return logits




#########----- 3. CREATE DATALOADER FOR LSTM TRAINING ---------- #######
import torch
from torch.utils.data import DataLoader

# === 1. Select GPU if available ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 2. Define collate function for variable-length LSTM batches ===
def lstm_collate_fn(batch):
    """
    Custom collate function for LSTM input with both sequence and static features.
    - Splits each batch element into sequence, static, label, and length tensors.
    - Separates PAGE_ID (categorical) from other continuous sequence features.
    - Moves all tensors to the specified device (GPU or CPU).
    """
    X_seq_batch, X_static_batch, y_batch, len_batch = zip(*batch)

    # Convert to stacked tensors
    X_seq_batch = torch.stack(X_seq_batch)
    X_static_tensor = torch.stack(X_static_batch)
    y_batch = torch.stack(y_batch)
    len_batch = torch.stack(len_batch)

    # Separate categorical page ID (index 0) and continuous features (index 1+)
    pageid_tensor = X_seq_batch[:, :, 0].long()
    features_tensor = X_seq_batch[:, :, 1:]

    # Move all tensors to device
    pageid_tensor = pageid_tensor.to(device)
    features_tensor = features_tensor.to(device)
    X_static_tensor = X_static_tensor.to(device)
    y_batch = y_batch.to(device)
    len_batch = len_batch.to(device)

    return pageid_tensor, features_tensor, X_static_tensor, y_batch, len_batch


# === 3. Define DataLoaders ===
BATCH_SIZE = 256

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lstm_collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lstm_collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lstm_collate_fn
)

print("DataLoaders created successfully.")


#########----- 4. TRAIN THE MODEL ---------- #######
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from datetime import datetime

# === settings ===
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 30
MODEL_SAVE_PATH = "loan_interest_lstm_best.pth"

# === instantiate model (example arguments kept short) ===
model = LoanInterestLSTM(
    embedding_size=128,
    seq_input_size=3,
    static_input_size=5,
    lstm_layers=2,
    dropout=0.3,
    hidden_size=128,
    dense_size=128,
    num_pages=len(pageid_dict)
)

# the loss function to binary cross-entropy losss( for binary classification)
criterion = nn.BCEWithLogitsLoss()
# updating a model's parameters(weights and biases) to minimize the loss after each training step
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
scaler = GradScaler()

model.to(device)

# === training loop ===
start_time = datetime.now()
train_losses, val_losses = [], []
best_val_auc = 0.0
wait = 0

#each epoch >> the model sees all the training data once 
#within each epoch, data is usually processed in mini-batches(32, 64 samples at a time)
for epoch in range(1, MAX_EPOCHS + 1):
    model.train() #set model to training mode
    epoch_train_losses = []
    all_train_probs, all_train_preds, all_train_labels = [], [], []

    for pageid_tensor, features_tensor, static_tensor, y_batch, len_batch in train_loader:
        optimizer.zero_grad() #clean gradient

        with autocast():
            logits = model(pageid_tensor, features_tensor, static_tensor, len_batch).squeeze(-1)
            #compute loss against true label
	    loss = criterion(logits, y_batch)

        # backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
	# update weights
	scaler.step(optimizer)
        scaler.update()

        epoch_train_losses.append(loss.item())

        probs = torch.sigmoid(logits).detach().cpu().numpy()
	preds = (probs > 0.5).astype(int)
        all_train_probs.extend(probs.tolist())
        all_train_preds.extend(preds.tolist())
        all_train_labels.extend(y_batch.detach().cpu().numpy().tolist())

    # training metrics
    train_loss = sum(epoch_train_losses) / max(1, len(epoch_train_losses))
    train_auc = roc_auc_score(all_train_labels, all_train_probs) if len(set(all_train_labels)) > 1 else float("nan")
    train_pr_auc = average_precision_score(all_train_labels, all_train_probs)



    # === validation ===
    model.eval()
    epoch_val_losses = []
    all_val_probs, all_val_labels, all_val_preds = [], [], []

    #not to track computations used to calculate gradients>> save memory and speed up evaluations
    with torch.no_grad():
        for pageid_tensor, features_tensor, static_tensor, y_batch, len_batch in val_loader:

            with autocast():
                logits = model(pageid_tensor, features_tensor, static_tensor, len_batch).squeeze(-1)
                loss = criterion(logits, y_batch)

            epoch_val_losses.append(loss.item())
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_val_probs.extend(probs.tolist())
            all_val_preds.extend(preds.tolist())
            all_val_labels.extend(y_batch.detach().cpu().numpy().tolist())

    val_loss = sum(epoch_val_losses) / max(1, len(epoch_val_losses))
    val_auc = roc_auc_score(all_val_labels, all_val_probs) if len(set(all_val_labels)) > 1 else float("nan")
    val_pr_auc = average_precision_score(all_val_labels, all_val_probs)

    # scheduler step
    scheduler.step(val_loss)

    # early stopping & model saving
      if val_auc > best_val_auc:
          best_val_auc = val_auc
          wait = 0
          torch.save(model.state_dict(), MODEL_SAVE_PATH)
          print(f"[Epoch {epoch}] Saved best model (val_auc={val_auc:.4f})")
      else:
          wait += 1
          if wait >= PATIENCE:
             print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
             break


    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # print summary per epoch
    print(
        f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
        f"train_auc={train_auc:.4f} | val_auc={val_auc:.4f} | "
        f"train_pr={train_pr_auc:.4f} | val_pr={val_pr_auc:.4f}"
    )
    print("val_classification_report:\n", classification_report(all_val_labels, all_val_preds, digits=4))



# === plot losses ===
plt.plot(train_losses, label="train_loss")
plt.plot(val_losses, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.show()

end_time = datetime.now()
print("Start:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
print("End:  ", end_time.strftime("%Y-%m-%d %H:%M:%S"))
print("Elapsed:", end_time - start_time)





# === final test evaluation ===
all_test_probs, all_test_preds, all_test_labels = [], [], []
model.eval()
with torch.no_grad():
    for pageid_tensor, features_tensor, static_tensor, y_batch, len_batch in test_loader:

	with autocast():
        	logits = model(pageid_tensor, features_tensor, static_tensor, len_batch).squeeze(-1)
        
	probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_test_probs.extend(probs.tolist())
        all_test_preds.extend(preds.tolist())
        all_test_labels.extend(y_batch.detach().cpu().numpy().tolist())

print("Test classification report:\n", classification_report(all_test_labels, all_test_preds, digits=4))
print("Test AUC:", roc_auc_score(all_test_labels, all_test_probs))
print("Test PR-AUC:", average_precision_score(all_test_labels, all_test_probs))


import pickle
with open ("scaler_seq.pkl", "wb") as f:
    pickle.dump(scaler_seq, f)
with open ("scaler_static.pkl", "wb") as f:
    pickle.dump(scaler_static, f)
