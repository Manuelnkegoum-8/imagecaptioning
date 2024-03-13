from .attention import *
from .embeddings import *



def create_mask_decoder(target,padding_idx=0):
    seq_len = target.size(1)
    pad_mask = (target != padding_idx).unsqueeze(1).unsqueeze(2)
    seq_mask = torch.tril(torch.ones(seq_len, seq_len)).to(target.device)
    trg_mask = pad_mask * seq_mask
    return trg_mask


class feedforward(nn.Module):
    def __init__(self,embed_dim=512,ff_hidden_dim=1024,dropout_rate = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
        )
    def forward(self, x):
        outputs = self.mlp(x)
        return outputs


class Encoder(nn.Module):
    def __init__(self, embed_dim=512, depth=4, heads=2, ff_hidden_dim=1024, dropout_rate = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiHeadSelfAttention(embed_dim,heads),
                nn.LayerNorm(embed_dim),
                feedforward(embed_dim,ff_hidden_dim, dropout_rate),
                nn.LayerNorm(embed_dim),
            ]))
    def forward(self, source,target=None,mask=None):
        x = source.clone()
        for attn,norm, ff,norm2 in self.layers:
            x = attn(Q=x,K=x,V=x,attn_mask = mask) + x
            x = norm(x)
            x = ff(x) + x
            x = norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embed_dim=512, depth=4, heads=2, ff_hidden_dim=1024, dropout_rate = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiHeadSelfAttention(embed_dim,heads), # masked self attention
                nn.LayerNorm(embed_dim),
                MultiHeadSelfAttention(embed_dim,heads), # cross atention
                nn.LayerNorm(embed_dim),
                feedforward(embed_dim,ff_hidden_dim, dropout_rate), # feed forward
                nn.LayerNorm(embed_dim)
            ]))
    def forward(self,encoder_out,decoder_in,mask=None):
        x = decoder_in.clone()
        for self_attn,norm1,cross_attn,norm2, ff,norm3 in self.layers:
            x = self_attn(Q=x,K=x,V=x,attn_mask = mask) + x
            x = norm1(x)
            x = cross_attn(Q=x,V=encoder_out,K=encoder_out) + x
            x = norm2(x)
            x = ff(x) + x
            x = norm3(x)
        return x

class Transformer(nn.Module):
    def __init__(self,height=224,
                width=224,
                n_channels=3,
                patch_size=16,
                dim=512,
                encoder_head=2,
                encoder_feed_forward=1024,
                encoder_depth=2,
                decoder_head=2,
                decoder_feed_forward=1024,
                decoder_depth=1,
                targ_len = 10,
                vocab_size = 100,
                padding_idx = 0):

        super().__init__()
        self.image_embedding = patch_embedding(height,width,n_channels,patch_size,dim)
        self.decoder_embedding = nn.Embedding(vocab_size,dim, padding_idx=padding_idx)
        self.decoder_positional = PositionalEncoding(dim, targ_len)
        self.n_patchs = height*width//(patch_size**2)
        # Create a diagonal attention mask
        self.diag_attn_mask = ~torch.eye(self.n_patchs, dtype=torch.bool)
        self.encoder = Encoder(dim,encoder_depth, encoder_head,encoder_feed_forward)
        self.decoder = Decoder(dim,decoder_depth, decoder_head,decoder_feed_forward)
        self.padding_idx = padding_idx
        self.fc = nn.Linear(dim,vocab_size)

    def encode(self,image):
        device_ = image.device
        x = self.image_embedding(image)
        out = self.encoder(x,x,mask=self.diag_attn_mask.to(device_))
        return out


    def decode(self,encoder_out,text):
        device_ = encoder_out.device
        decoder_mask = create_mask_decoder(text,padding_idx=self.padding_idx)
        y = self.decoder_positional(self.decoder_embedding(text))
        out = self.decoder(encoder_out,y,mask=decoder_mask.to(device_))

        return self.fc(out)
        
    def forward(self, image, text):
        encoder_out = self.encode(image)
        outputs = self.decode(encoder_out,text)
        return outputs


@torch.no_grad()
def greedy_decoding(model, image, max_len, start_idx,end_idx):
    model.eval()
    decoder_input = torch.ones(1, 1, dtype=torch.long) * start_idx
    decoded_outputs = decoder_input
    decoded_outputs = decoded_outputs.to(image.device)
    encoder_out = model.encode(image)

    for i in range(max_len):
        decoder_output = model.decode(encoder_out,decoded_outputs)  # Decode next token
        next_word = decoder_output[:, -1].argmax(dim=-1).unsqueeze(-1)  # Get most probable word index
        if next_word==end_idx:
            break
        decoded_outputs = torch.cat([decoded_outputs, next_word], dim=-1)  # Append next word to output sequence
        
    return decoded_outputs[:, 1:] 