import torch
import torch.nn.functional as F
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import json
import matplotlib.pyplot as plt

# 初始化tokenizer
tokenizer_path = ("E:\\pythonProject\\AI_Course_Transformers\\AI_Course_Transformers\\meta-llama\\"
                  "Meta-Llama-3-8B-Instruct\\original\\tokenizer.model")
special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

# print(tokenizer.encode("hello world!"))
# print(tokenizer.decode(tokenizer.encode("hello world!")))

# 加载模型
model = torch.load("E:\\pythonProject\\AI_Course_Transformers\\AI_Course_Transformers\\meta-llama\\"
                   "Meta-Llama-3-8B-Instruct\\original\\consolidated.00.pth")

# print(json.dumps(list(model.keys())[:20], indent=4))

# 加载配置
with open("E:\\pythonProject\\AI_Course_Transformers\\AI_Course_Transformers\\"
          "meta-llama\\Meta-Llama-3-8B-Instruct\\original\\params.json", "r") as f:
    config = json.load(f)

class Llama3Model(torch.nn.Module):
    def __init__(self, config, model_weights, device):
        super(Llama3Model, self).__init__()
        self.dim = config["dim"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.vocab_size = config["vocab_size"]
        self.multiple_of = config["multiple_of"]
        self.ffn_dim_multiplier = config["ffn_dim_multiplier"]
        self.norm_eps = config["norm_eps"]
        self.rope_theta = torch.tensor(config["rope_theta"], device=device)

        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.dim).to(device)
        self.embedding_layer.weight.data.copy_(model_weights["tok_embeddings.weight"].to(device))

        # 初始化其他层和权重并移动到设备
        self.attention_norm_weights = [model_weights[f"layers.{i}.attention_norm.weight"].to(device) for i in range(self.n_layers)]
        self.ffn_norm_weights = [model_weights[f"layers.{i}.ffn_norm.weight"].to(device) for i in range(self.n_layers)]
        self.wq_weights = [model_weights[f"layers.{i}.attention.wq.weight"].to(device) for i in range(self.n_layers)]
        self.wk_weights = [model_weights[f"layers.{i}.attention.wk.weight"].to(device) for i in range(self.n_layers)]
        self.wv_weights = [model_weights[f"layers.{i}.attention.wv.weight"].to(device) for i in range(self.n_layers)]
        self.wo_weights = [model_weights[f"layers.{i}.attention.wo.weight"].to(device) for i in range(self.n_layers)]
        self.w1_weights = [model_weights[f"layers.{i}.feed_forward.w1.weight"].to(device) for i in range(self.n_layers)]
        self.w2_weights = [model_weights[f"layers.{i}.feed_forward.w2.weight"].to(device) for i in range(self.n_layers)]
        self.w3_weights = [model_weights[f"layers.{i}.feed_forward.w3.weight"].to(device) for i in range(self.n_layers)]
        self.final_norm_weight = model_weights["norm.weight"].to(device)
        self.output_weight = model_weights["output.weight"].to(device)

    def rms_norm(self, tensor, norm_weights):
        return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + self.norm_eps)) * norm_weights

    def forward(self, tokens):
        token_embeddings_unnormalized = self.embedding_layer(tokens).to(torch.bfloat16)
        final_embedding = token_embeddings_unnormalized

        freqs_cis = self.compute_freqs_cis(len(tokens))

        for layer in range(self.n_layers):
            qkv_attention_store = []
            layer_embedding_norm = self.rms_norm(final_embedding, self.attention_norm_weights[layer])
            q_layer = self.wq_weights[layer].view(self.n_heads, self.wq_weights[layer].shape[0] // self.n_heads, self.dim)
            k_layer = self.wk_weights[layer].view(self.n_kv_heads, self.wk_weights[layer].shape[0] // self.n_kv_heads, self.dim)
            v_layer = self.wv_weights[layer].view(self.n_kv_heads, self.wv_weights[layer].shape[0] // self.n_kv_heads, self.dim)
            w_layer = self.wo_weights[layer]

            for head in range(self.n_heads):
                q_layer_head = q_layer[head]
                k_layer_head = k_layer[head // 4]
                v_layer_head = v_layer[head // 4]
                q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
                k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
                v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)

                q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
                q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
                q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
                q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

                k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
                k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
                k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
                k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

                qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (128) ** 0.5
                mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=1)
                qk_per_token_after_masking = qk_per_token + mask
                qk_per_token_after_masking_after_softmax = F.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
                qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
                qkv_attention_store.append(qkv_attention)

            stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
            embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
            embedding_after_edit = final_embedding + embedding_delta
            embedding_after_edit_normalized = self.rms_norm(embedding_after_edit, self.ffn_norm_weights[layer])
            w1 = self.w1_weights[layer]
            w2 = self.w2_weights[layer]
            w3 = self.w3_weights[layer]
            output_after_feedforward = torch.matmul(F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
            final_embedding = embedding_after_edit + output_after_feedforward

        final_embedding = self.rms_norm(final_embedding, self.final_norm_weight)
        logits = torch.matmul(final_embedding[-1], self.output_weight.T)
        return logits

    def compute_freqs_cis(self, seq_length):
        zero_to_one_split_into_64_parts = torch.tensor(range(64), device=self.rope_theta.device) / 64
        freqs = 1.0 / (self.rope_theta ** zero_to_one_split_into_64_parts)
        freqs_for_each_token = torch.outer(torch.arange(seq_length, device=self.rope_theta.device), freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
        return freqs_cis

# 检查是否有CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型实例并移动到CUDA设备
llama_model = Llama3Model(config, model, device).to(device)

# 示例输入
tokens = torch.tensor([128000] + tokenizer.encode("the answer to the ultimate question of life, the universe, and everything is ")).to(device)

# 获取模型输出
logits = llama_model(tokens)

# 打印logits的形状
print(logits)

# # 获取每个位置的预测词索引
predicted_tokens = torch.argmax(logits, dim=-1)
print(predicted_tokens)
print(tokenizer.decode([predicted_tokens.item()]))

# # 使用tokenizer将词索引解码成对应的词
# predicted_text = tokenizer.decode(predicted_tokens.cpu().tolist())
# print(predicted_text)

