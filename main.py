import time
import torch
import torch.nn as nn
import tiktoken
import os
import urllib.request
from torch.utils.data import Dataset, DataLoader


GPT_CONFIG_124M = {
    "vocab_size": 50257,    #Vocabulary Size
    "context_length": 256, #Context length <==== Adjusted for training simplicity (1024 = GPT2)
    "emb_dim": 768,         #Embedding dimensions
    "n_heads": 12,          #Number of attention heads
    "n_layers": 12,         #Number of layers
    "drop_rate": 0.1,       #Dropout rate
    "qkv_bias": False       #Query-Key-Value bias
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Uncommenting the following lines will allow the code to run on Apple Silicon chips instead cpu
#Note: Resulting loss values may be slightly different

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
    
print(f"Using {device} device: {torch.version.cuda}")


#Instance tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
#Default start_context => Used in testing
start_context = "Every effort moves you"

#------------------------Loads Short-story for testing-----------------------------
file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode("utf-8")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)

else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print(f"Characters: {len(text_data)}, Tokens: {len(tokenizer.encode(text_data))}")
#----------------------------------------------------------------------------------

#Takes raw text, encodes it, and a batch dimension
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #Add batch dimension
    return encoded_tensor

#Takes batched token_ids (typically final_output) removes the batch dim and decodes to text
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) #Remove batch dimension
    return tokenizer.decode(flat.tolist())

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

#TRAINING VARIABLES:
#Train/Validation Ratio
train_ratio = 0.90
#Splits the text_data relative to training ratio => example: train_Data = 90%, val_data = 10%
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx] 
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size = 2, #Set very low, 2, due to demands processing large batches. GPT-2 = 1024
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last = True,
    shuffle = True,
    num_workers = 0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size = 2,  #Set very low, 2, due to demands processing large batches. GPT-2 = 1024
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last = False,
    shuffle = False,
    num_workers = 0
)      

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

#Instance model/optimizer
model = GPTModel(GPT_CONFIG_124M)
model.to(device) 

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

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

    #Main training loop
    for epoch in range(num_epochs):
        model.train() #Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration

            #Caclulates loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            #Backward propigation
            loss.backward() #Calculate loss gradient
            
            #Update model weights using loss gradients
            optimizer.step() 
            tokens_seen += input_batch.numel() #Returns the total num of tokens seen
            global_step += 1

            #Optional evaluation step => Runs every eval_frequence # of batches 
            #     Prints training and validation loss to terminal
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
    
    #Print a sample text (50 tokens) after each epoch => shows how well the model is performing
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    #Calculates loss for the training and validation dataloaders
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

#Calculates 50 token sample to show how well the model is performing. Called every epoch.
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

#-----------------------------Training Test------------------------------------------
#NOTE: Uncomment the following code to calculate execution time
start_time = time.time()

torch.cuda.empty_cache() #Added due to maxing out cache in GPU
torch.manual_seed(123) #Reproducibility

num_epochs = 10 # Go through entire dataset 10 times
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

#NOTE: Uncomment the following code to calculate execution time
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")                   

# #---------------------------------------------------------------------------------------------


# #Model parameter / optimizer data saving 

# torch.save({
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     },
#     "model_and_optimizer.pth"
# )

#---------------------------------------------------------------------------------------------
# #The stored value from the above saved code can then be loaded back into optimizer and model
# checkpoint = torch.load("model_and_optimizer.pth")
# model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.train(); #Model in train mode