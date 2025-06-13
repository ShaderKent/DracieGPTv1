import os
import time
import urllib
import json
import ssl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import partial
import numpy as np

from gpt_download3 import download_and_load_gpt2

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

GPT_CONFIG_124M = {
    "vocab_size": 50257,     #Vocabulary Size
    "context_length": 1024,  #Context length
    "emb_dim": 768,          #Embedding dimensions
    "n_heads": 12,           #Number of attention heads
    "n_layers": 12,          #Number of layers
    "drop_rate": 0.1,        #Dropout rate
    "qkv_bias": False        #Query-Key-Value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

checkpoint_load = "GPT-2_355M_Instructions.pth"
checkpoint_save = "GPT-2_355M_Instructions.pth"
requested_device = "cpu"
torch.manual_seed(123)

def instanceDevice(requested_device):
    #Instances device/prints currently used device => Apple compatibility options

    if(requested_device == "cuda"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    #print("Allocated Memory:", torch.cuda.memory_allocated(device)/ 1000000000)

    #----------------------------------------------------------------------------
    #Uncommenting the following lines will allow the code to run on Apple Silicon chips instead cpu
    #Note: Resulting loss values may be slightly different

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    #----------------------------------------------------------------------------
    
    print(f"Requested: {requested_device}, Using: {device}")
    return device

device = instanceDevice(requested_device)

#Default start_context => Used in testing
start_context = "Every effort moves you"

#Instance tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

#Takes raw text, encodes it, and a batch dimension
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #Add batch dimension
    return encoded_tensor

#Takes batched token_ids (typically final_output) removes the batch dim and decodes to text
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) #Remove batch dimension
    return tokenizer.decode(flat.tolist())

#Basic dataset class used in create_dataloader_v1 => used in pretraining
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    #Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    #Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    #Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, #If the last batch is shorter than context_length, ignore it
        num_workers=num_workers #Option for parallel computation
    )

    return dataloader
   
#Method to normalize layers, used 2x in transformer 1x on final output
#Normalization = values are adjusted to have a mean of 0 and a variance of 1
#    Prevents numbers from getting too big or two small, which would cause issues for learning
class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5 #A small constant which is added to the variance to prevent / 0 in calcs
        self.scale = nn.Parameter(torch.ones(emb_dim)) #Trainable param for this step(same dim)
        self.shift = nn.Parameter(torch.zeros(emb_dim)) #Trainable param for this step(same dim)

    def forward(self, x):
        #Calculate the mean of the cols (for each token, or row)
        mean = x.mean(dim=-1, keepdim=True)
        #Calculate the variance of the cols (for each token, or row)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        #Normalize each value in each row
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        #Return normalized values scaled (* by scale) and shifted (+ by shift)
        return self.scale * norm_x + self.shift

#Multi-head Attention module. Creates token_emb and pos_emb layers, running calculations to
#     Return context matrixies that hold positional and semantic meaning stored in multi dim
#     arrays for each token relative to other tokens. Calculations are multi headed in that
#     the intensive computations are split into different gpu 'heads' to allow calc in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out #DimOut of context vector, typically same as dimIn
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape #Input shape (batchSize, tokensInContext, dimOfContext)

        #Initialize traninable weight matrices
        keys = self.W_key(x) # Shape: (b, num_token, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        #Implicity split the matrix by adding a "num_heads" dimension
        #Unroll last dim: (b, num_tokens, d_out)) => (b, num_tokens, num_heads, head_dim)
        #Example: If batch = 1, num_tokens = 3, d_in = 6, and num_heads = 2:
        #        a keys of shape (1,3,6) will be converted to (1,3,2,3) <= d_in / num_heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        #Currently each tensor is grouped by the number of tokens, but for calc they need to 
        #     be grouped by the num_heads so they are transposed.
        #Transpose: (b, num_tokens, num_heads, head_dim) => (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        #Compute scaled dot-product attention (aka self-attention) with a causal mask
        #Keys needs to be transposed in the 3rd (num_tokens) + 4th(head_dim <=(num_dimPerHead))
        #attn_scores = queries(b,#heads,#tokens,head_dim)*keys(b,#heads,head_dim,#tokens)
        #     result = (b, #heads, #tokens, #tokens) => matrix of token to token relationships
        attn_scores = queries @ keys.transpose(2,3) # Dot product for each head 

        #Original mask truncated to number of tokens and converted to boolean
        #Prevents partial batchsizes (typically end of document) from causing errors
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        #Use the mask to fill attention scores after the current query
        # -inf prevents future values from influencing soft-max
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        #Softmax is applied => Each layer adds up to 1 (allows % to convey meaning)
        #sqrt(head_dim) of col is added to prevent vector multiplication from growing to large
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        #Dropout is applied
        attn_weights = self.dropout(attn_weights)

        #Context vector matrix is computed 
        #context_Vec = attn_weights(b,#heads,#tokens,#tokens)*values(b,#heads,#tokens,head_dim)
        #     result = (b, #heads, #tokens, head_dim)
        #Transposed to:(b, #tokens, #heads, head_dim) => In preparation for merging last two dim
        context_vec = (attn_weights @ values).transpose(1,2)

        #Combine heads, where self.d_out = self.num_heads * self.head_dim ('roll-back' to prev)
        #Example: If batch = 1, num_tokens = 3, d_in = 6, and num_heads = 2; d_in / #heads prev
        #     (1,3,2,3) + (1,3,2,3) => (1,3,6) where output = (b, #tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec #Shape: (b,#token,d_out)

#GELU is a method of neuron activation. In essence, values less than 0 are set to ~0, and values
#     above zero are linear. This method is called in the feed forward block (after x4 dim)
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
#Feed forward: Expands dim x4 then calls GELU and finally contracts dims to perserve dim size
#     This allows stacking of nn, with expansion allowing for more nuanced connections btwn toks
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #nn.Sequential is a method that allows the chaining of multiple nn together
        #Construction of nn
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), #Expansion
            GELU(),                                        #Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), #Contraction
        )

    def forward(self, x):
        return self.layers(x)

#Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        #ATTENTION BLOCK
        #Shortcut connection for attention block
        #     -'Shortcut' outputs by adding the input to the output => prevents gradient decay 
        shortcut = x
        #Layer Normalization 1
        x = self.norm1(x)
        #Self Attention Mechanism call => multihead attention
        x = self.att(x) # Shape[batch_size, num_tokens, emb_size]
        #Dropout 1
        #Dropout randomly deactivates a number of the values in the nn to prevent nodes from 
        #    getting 'lazy'. Since they are forced to function due to the absence of others
        #    their values are more consistently refreshed, increasing accurance of the model
        x = self.drop_shortcut(x)
        x = x + shortcut # Add input to output

        #FEED FORWARD BLOCK
        #Shortcut connection for feed foward block
        shortcut = x
        #Layer normalization 2
        x = self.norm2(x)
        #Feed forward call
        x = self.ff(x) 
        #Dropout 2
        x = self.drop_shortcut(x)
        x = x + shortcut # Add input ot output

        return x

#GPT Model (Core)
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #Generates the token_emb matrix [vocab_size = 50257, emb_dim = 768]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        #Generates the positional_emb matrix [context_length = 1024, emb_dim = 768]
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        #Generates dropout layer
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        #Generates a sequential set of TransformerBlock objects of # (n_layers = 12) length
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        #Generates (final) normalization layer
        self.final_norm = LayerNorm(cfg["emb_dim"])

        #Generates the final output nn of size[emb_dim = 768, vocab_size = 50257]
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        #Gets input and sets batch_size and seq_len = to the passed in values # and length
        batch_size, seq_len = in_idx.shape
        #Creates a subMatrix of values from tok_emb matrix of size [#ofTokens, emb_dim = 768]
        tok_embeds = self.tok_emb(in_idx)
        #Creates a subMatrix of values from pos_emb matrix of size [#ofTokens, emb_dim = 768]
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        #Adds the tok_embeds (semantic meaning) + pos_embeds (positional meaning)
        x = tok_embeds + pos_embeds #Shape[batch_size, num_tokens, emb_size]
        #Applies inital dropoff
        x = self.drop_emb(x)
        #Passes values/input into the transformer blocks (#of Transformers = n_layers = 12)
        x = self.trf_blocks(x)
        #Normalizes final output (final normalization layer)
        x = self.final_norm(x)
        #Passes values/input into final nn to get a tensor of probs for each vocab (logits)
        logits = self.out_head(x)
        return logits #Size = [batch_size, num_tokens, vocab_size = 50257]

# Instance model/optimizer => Used when generating a mew state (fresh)
def fresh_untrained_model():
    model = GPTModel(GPT_CONFIG_124M)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    model.to(device)
    return model, optimizer

#Text/token generation method
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    #Get logits tensor by passing input into model and selectio on the last val (next prediction)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        #Optional top-k sampling => helps prevents overfitting
        if top_k is not None:
            #Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k) #Returns positions of selected tokens
            min_val = top_logits[:, -1] #Select all tokens less probable than the top-k tokens
               #replace all selected values with -inf to prevent softmax influence => to device
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        #Optional temperature scaling => helps prevent overfitting
        if temperature > 0.0:
            logits = logits / temperature 
            #Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1) # Shape[batch_size, context_length]
            #Sample from the distribution => gets selected output token
            idx_next = torch.multinomial(probs, num_samples=1) # [batch_size, 1]
        
        #When temperature scaling is disabled
        else:
        #Select output token => token with highest probability
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) # [batch_size, 1]

        if idx_next == eos_id: #Stop generating early if end-of-sequence token is encountered
            break

        #Append output token to the current input token sequence
        idx = torch.cat((idx, idx_next), dim=1) # [batch_size, num_tokens + 1]
    
    return idx
    
#Calculates loss for a single batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    
    #CALCULATE CROSS ENTROPY LOSS => 'measures difference between 2 prob. distributions'
    #Step 1.1: Look up target tokens on the probability tensor and get associated probability
    #        => Shape[batch_size, num_target_tokens, currentprobability]
    #     1.2: Merge values into a single array => [b1.num1.currentP, b2.num2...]
    #        => Result = logits_flat
    #Step 2: Calculate the cross entropy loss of those values by:
    #     2.1: Take log of values => [log(Val1), log(Val2), log(Val3)...]
    #     2.2: Average values => [Sum(result1, result2, result3...)/#ofResults]
    #     2.3: Get negative => -[value] => 'Loss' or 'cross entropy loss'
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

#Calculates average loss for all batches 
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    #Handles exception => loader has no data
    if len(data_loader) == 0:
        return float("nan")

    #If a specific num_batches is not provided, set the num batches = dataloader's length
    elif num_batches is None:
        num_batches = len(data_loader)
        
    else:
    #If num_batches exceeds the number of batches in the dataloader:
    #   Reduce the number of batches to match the total number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
        
    #for each batch calculate the loss, then add those losses together
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches #Final total is averaged by dividing by num_batches
    
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    #Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    average1kTrain, average1kVal = 0, 0

    #Main training loop
    for epoch in range(num_epochs):
        model.train() #Set model to training mode

        for input_batch, target_batch in train_loader:
            torch.cuda.empty_cache()

            optimizer.zero_grad() # Reset loss gradients from previous batch iteration

            #Caclulates loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            #Backward propigation
            loss.backward() #Calculate loss gradient
            
            #Update model weights using loss gradients
            optimizer.step() 
            tokens_seen += input_batch.numel() #Returns the total num of tokens seen
            global_step += 1

            #Optional GPU Memory Allocated Printout
            # print("Allocated Memory:", torch.cuda.memory_allocated(device)/ 1000000000)

            #Optional evaluation step => Runs every eval_frequence # of batches 
            #     Prints training and validation loss to terminal
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                average1kTrain = average1kTrain + train_loss
                average1kVal = average1kVal + val_loss
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            if global_step % (eval_freq*3) == 0:
                generate_and_print_sample(
                    model, tokenizer, device, start_context
                )
    
    #Print a sample text (50 tokens) after each epoch => shows how well the model is performing
        
    return train_losses, val_losses, track_tokens_seen

#Calculates loss for the training and validation dataloaders (full)
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

#Calculates 50 token sample to show how well the model is performing.
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    #Context_size is set to current #tokens in pos_emb
    context_size = model.pos_emb.weight.shape[0] 
    encoded = text_to_token_ids(start_context, tokenizer).to(device)  #Encodes input=>to device
    
    #Takes a seed text (start_context) and generates new tokens
    with torch.no_grad():
        # token_ids = generate_text_simple(
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size,
            temperature=1.4, top_k=5
        )    
    decoded_text = token_ids_to_text(token_ids, tokenizer) #Decodes result
    print(decoded_text.replace("\n", " ")) #Compact print format
    model.train()

#Small utility function that checks whether two tensors or arrays are the same dim shape.
#If true, return 'right', or assign the right value, if false throw error
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
    
#Divides pretrained gpt data and assigns it to correct parameters
def load_weights_into_gpt(gpt, params):
    #Set token_emb and pos_emb = to the trained weights for those matrixies
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    #For each block (multi-headed attention transformer block)
    for b in range(len(params["blocks"])):
        #Split the combined (c_attn.w) attn matrix into respective q_w, k_w, v_w and assign
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        #Splits the combined bias (c_attn.b) matrix into respective q_b, k_b, v_b and assign
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        #Assigns output projection layer weight and bias for fully connected and proj layer
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        #Assign feed forward nn weights and bias
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        #Assign layer normalization scale and shift values for layer_norm1 and layer_norm2
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    #Sets scale and shift for final normalization layer
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

    #To limit computational demands, gpt-2 uses 'weight-tying', using the same tok emb matrix
    #    for both the main tok_emb and final output dim
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

    print(f"{CHOOSE_MODEL} weights assigned")

#Downloads a 'fresh' instance of pretrained gpt2 parameters of "model_size" (small, medium, large, xl)
def download_fresh_gpt_pretrained():
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.to(device)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    return model, optimizer

#Loads previously trained state
def load_checkpoint(desired_checkpoint_path):
    checkpoint = torch.load(desired_checkpoint_path, weights_only=True)

    model = GPTModel(BASE_CONFIG)
    model.to(device) 
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train(); #Model in train mode
   
    print(f"Checkpoint: '{desired_checkpoint_path}' loaded")
    return model, optimizer

#Model parameter / optimizer data saving 
def save_checkpoint(desired_checkpoint_path, model, optimizer):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        },
        f"{desired_checkpoint_path}")
    print(f"Checkpoint: '{desired_checkpoint_path}' saved")


    
#Utility to download json formatted data with a given url => returns file's data
def download_and_load_file(file_path, url):
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode - ssl.CERT_NONE

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url, context=ssl_context) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("[" + text_data + "]")
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

#Converting Instructions into Alpaca format
def format_input_alpaca(instruction, input):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
    )

    input_text = f"\n\n### Input:\n{input}" if input else ""

    return instruction_text + input_text

#Dataset formatted for alpaca instructions based data
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        #Pre-tokenize text
        self.encoded_texts = []
        for entry in data:
            instruction = entry["instruction"]
            input = entry["input"]
            output = entry["output"]

            instruction_plus_input = format_input_alpaca(instruction, input)
            response_text = f"\n\n### Response: \n{output}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

#Creates input target pairs, using padding to standardize length to largest input in batch
def custom_collate_fn(
    batch, 
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    #Find the longest sequence in the batch and increase the max length by +1 which will add 1 more padTok
    batch_max_length = max(len(item)+1 for item in batch)

    #Pad and prepare inputs
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        #Add an <|endoftext|> token (50256)
        new_item += [pad_token_id]
        #Pad sequences to batch_max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        #Adjust added padded token as needed for targets/inputs
        inputs = torch.tensor(padded[:-1]) #Truncate the last token for inputs
        targets = torch.tensor(padded[1:]) #Shift +1 to the right for targets

        #Replace all but the first paddingToken with ignoreIndex (-100)
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
            
        inputs_lst.append(inputs)
        targets_lst.append(targets)
        
    #Convert list of inputs to tensor and transfer to target device    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    
    return inputs_tensor, targets_tensor   

#Customized collate_fn call
customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

#Generates dataloaders using Alpaca instruction data for finetuning an instruction based model
def generate_alpaca_dataloaders():
    import alpaca_full
    # data = alpaca_full.alpaca_data[:1000]
    data = alpaca_full.alpaca_data[1001:2000]
    # data = alpaca_full.alpaca_data[2001:3000]
    # data = alpaca_full.alpaca_data[3001:4000]
    # data = alpaca_full.alpaca_data[4001:5000]
    # data = alpaca_full.alpaca_data[5001:6000]
    # data = alpaca_full.alpaca_data[6001:7000]
    # data = alpaca_full.alpaca_data[7001:8000]
    # data = alpaca_full.alpaca_data[8001:9000]
    # data = alpaca_full.alpaca_data[9001:10000]
    # data = alpaca_full.alpaca_data[10001:11000]
    # data = alpaca_full.alpaca_data[11001:12000]
    # data = alpaca_full.alpaca_data[12001:13000]
    # data = alpaca_full.alpaca_data[13001:14000]
    # data = alpaca_full.alpaca_data[14001:15000]


    #Split Dataset into Train-Test-Validation
    train_portion = int(len(data) * 0.85) #85% for training
    test_portion = int(len(data) * 0.1) #10% for training
    val_portion = int(len(data)) - train_portion - test_portion #Remaining 5% for validation

    train_data = data[:train_portion]
    test_data =  data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Train: ", len(train_data), "Test: ", len(test_data), "Val", len(val_data))
    num_workers = 0
    batch_size = 1

    #Create train dataset => using InstructionDataset class and passing it into Dataloader
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        collate_fn = customized_collate_fn,
        shuffle = True,
        drop_last = True,
        num_workers = num_workers
    )
    print("Training Dataset Created")

    #Create validation dataset => using InstructionDataset class and passing it into Dataloader
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        collate_fn = customized_collate_fn,
        shuffle = False,
        drop_last = False,
        num_workers = num_workers
    )
    print("Validation Dataset Created")

    #Create test dataset => using InstructionDataset class and passing it into Dataloader
    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        collate_fn = customized_collate_fn,
        shuffle = False,
        drop_last = False,
        num_workers = num_workers
    )
    print("Test Dataset Created")
    return(train_loader, val_loader, test_loader, val_data)

#Trains using Alpaca formatted dataloaders, used to finetune a pretrained model with instructions.
def train_on_alpaca ():

    train_loader, val_loader, test_loader, val_data = generate_alpaca_dataloaders()
    start_time = time.time()

    num_epochs = 1
    start_context_instruction = val_data[3]["instruction"]
    start_context_input = val_data[3]["input"]

    if device == "cuda": 
        torch.cuda.empty_cache() #Added due to maxing out cache in GPU
        print("Allocated Memory:", torch.cuda.memory_allocated(device)/ 1000000000)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input_alpaca(start_context_instruction, start_context_input), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

#Basic query using Alpaca instruction format (use with finetuned Alpaca model)
def query_alpaca(instruction, input, new_tokens, context_size, temperature, top_k):
    formatted_text = format_input_alpaca(instruction, input)
    query = text_to_token_ids(formatted_text, tokenizer)
    token_ids = generate(model, query, new_tokens, context_size=context_size, temperature=temperature, top_k=top_k)
    result = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        result[len(formatted_text):]
        .replace("### Response:", "")
        .strip()
    )
    print(response_text)
#-----------Run------------
model, optimizer = load_checkpoint(checkpoint_load)

#RUN ALPACA TRAINING
# train_on_alpaca()

#RUN QUERY ON ALPACA FINETUNED MODEL
test_instruction = "What is a API?"
test_input = ""
query_alpaca(test_instruction, test_input, 100, 1024, 1.4, 10)

#----------Save------------
# save_checkpoint(checkpoint_save, model, optimizer)
