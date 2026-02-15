"""
Task 4: Text Similarity Web Application
Uses the custom S-BERT model from Task 2 to predict NLI labels
(entailment, neutral, contradiction) and compute cosine similarity.
"""

import os
import sys
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# Device setup
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ────────────────────────────────────────────────────────────────────────────
# Model architecture (same as Task 1 & Task 2)
# ─────────────────────────────────────────────────────────────────────────────
class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, n_segments, d_model, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(self.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


def get_attn_pad_mask(seq_q, seq_k, device):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).to(device)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, device):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(device)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, device):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(self.d_k, self.device)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = nn.Linear(self.n_heads * self.d_v, self.d_model, device=self.device)(context)
        return nn.LayerNorm(self.d_model, device=self.device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, device)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class BERT(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_ff, d_k, n_segments, vocab_size, max_len, device):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, max_len, n_segments, d_model, device)
        self.layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, d_k, device) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        self.device = device

    def get_last_hidden_state(self, input_ids, segment_ids):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, _ = layer(output, enc_self_attn_mask)
        return output


# ─────────────────────────────────────────────────────────────────────────────
# Load vocab and model
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_PATH = os.path.join(os.path.dirname(__file__), '..', 'word2id.json')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'bert_model_with_config.pth')
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), '..', 'classifier_head.pth')

# Load vocabulary
with open(VOCAB_PATH, 'r') as f:
    word2id = json.load(f)
id2word = {int(v): k for k, v in word2id.items()}
vocab_size = len(word2id)
print(f"Loaded vocab: {vocab_size} words")

# Load BERT model with config
checkpoint = torch.load(MODEL_PATH, map_location=device)
cfg = checkpoint['config']

model = BERT(
    cfg['n_layers'], cfg['n_heads'], cfg['d_model'], cfg['d_ff'], cfg['d_k'],
    cfg['n_segments'], vocab_size, cfg['max_len'], device
).to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
print(f"Loaded BERT: n_layers={cfg['n_layers']}, n_heads={cfg['n_heads']}, d_model={cfg['d_model']}")

# Load classifier head
d_model = cfg['d_model']
max_len = cfg['max_len']
classifier_head = nn.Sequential(
    nn.Linear(d_model * 3, d_model),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_model, 3)
).to(device)
if os.path.exists(CLASSIFIER_PATH):
    classifier_head.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    print("Loaded classifier head from classifier_head.pth")
else:
    print("WARNING: classifier_head.pth not found, using random weights")
classifier_head.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer and pooling (same as Task 2)
# ─────────────────────────────────────────────────────────────────────────────
def custom_tokenize(text_input, max_length=128):
    if isinstance(text_input, str):
        text_input = [text_input]
    all_input_ids = []
    all_attention_masks = []
    for text in text_input:
        cleaned = text.lower()
        cleaned = re.sub("[.,!?\\\\-]", '', cleaned)
        words = cleaned.split()
        tokens = [word2id['[CLS]']]
        for w in words:
            tokens.append(word2id.get(w, word2id['[MASK]']))
        tokens.append(word2id['[SEP]'])
        if len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [word2id['[SEP]']]
        attention_mask = [1] * len(tokens)
        pad_len = max_length - len(tokens)
        tokens.extend([0] * pad_len)
        attention_mask.extend([0] * pad_len)
        all_input_ids.append(tokens)
        all_attention_masks.append(attention_mask)
    return {'input_ids': all_input_ids, 'attention_mask': all_attention_masks}


def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool


def predict_nli(premise, hypothesis):
    """Predict NLI label and cosine similarity between premise and hypothesis."""
    with torch.no_grad():
        inputs_a = custom_tokenize(premise, max_length=max_len)
        inputs_b = custom_tokenize(hypothesis, max_length=max_len)

        ids_a  = torch.LongTensor(inputs_a['input_ids']).to(device)
        attn_a = torch.LongTensor(inputs_a['attention_mask']).to(device)
        ids_b  = torch.LongTensor(inputs_b['input_ids']).to(device)
        attn_b = torch.LongTensor(inputs_b['attention_mask']).to(device)
        seg_a  = torch.zeros_like(ids_a).to(device)
        seg_b  = torch.zeros_like(ids_b).to(device)

        u_hidden = model.get_last_hidden_state(ids_a, seg_a)
        v_hidden = model.get_last_hidden_state(ids_b, seg_b)

        u = mean_pool(u_hidden, attn_a)
        v = mean_pool(v_hidden, attn_b)

        # NLI classification: softmax(W^T * (u, v, |u-v|))
        uv_abs = torch.abs(torch.sub(u, v))
        x = torch.cat([u, v, uv_abs], dim=-1)
        logits = classifier_head(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))

        # Cosine similarity
        u_np = u.cpu().numpy().reshape(1, -1)
        v_np = v.cpu().numpy().reshape(1, -1)
        cos_sim = float(sklearn_cosine_similarity(u_np, v_np)[0, 0])

    label_names = ['Entailment', 'Neutral', 'Contradiction']
    return {
        'label': label_names[pred_label],
        'confidence': {
            'entailment': float(probs[0]),
            'neutral': float(probs[1]),
            'contradiction': float(probs[2])
        },
        'cosine_similarity': cos_sim
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    premise = data.get('premise', '').strip()
    hypothesis = data.get('hypothesis', '').strip()

    if not premise or not hypothesis:
        return jsonify({'error': 'Both premise and hypothesis are required.'}), 400

    result = predict_nli(premise, hypothesis)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)