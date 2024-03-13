import numpy as np
import random,json
import torch
from PIL import Image
from transformer.model import*
import torchvision.transforms as transforms
import torch
import gradio as gr
import argparse
import torchvision.transforms as T


parser = argparse.ArgumentParser(description='Image Captioning inference script')

# Data args
parser.add_argument('--max_seq_len', default=60, type=int, help='max sequence length')

# Model parameters
parser.add_argument('--height', default=32, type=int, metavar='N', help='image height')
parser.add_argument('--width', default=32, type=int, metavar='N', help='image width')
parser.add_argument('--channel', default=3, type=int, help='disable cuda')
parser.add_argument('--enc_heads', default=12, type=int, help='number of encoder  heads')
parser.add_argument('--enc_depth', default=9, type=int, help='number of encoder blocks')
parser.add_argument('--dec_heads', default=12, type=int, help='number of decoder  heads')
parser.add_argument('--dec_depth', default=1, type=int, help='number of decoder blocks')
parser.add_argument('--patch_size', default=4, type=int, help='patch size')
parser.add_argument('--dim', default=192, type=int, help='embedding dim of patch')
parser.add_argument('--enc_mlp_dim', default=384, type=int, help='feed forward hidden_dim for an encoder block')
parser.add_argument('--dec_mlp_dim', default=384, type=int, help='feed forward hidden_dim for a decoder block')




args = parser.parse_args()
height, width, n_channels = args.height, args.width, args.channel
patch_size, dim, enc_head = args.patch_size, args.dim, args.enc_heads
enc_feed_forward, enc_depth = args.enc_mlp_dim, args.enc_depth
dec_feed_forward, dec_depth = args.dec_mlp_dim, args.dec_depth
dec_head = args.dec_heads


# to load the dictionary from the JSON file
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transforms =  T.Compose([
                        T.Resize((height,width)),
                        T.ToTensor()
                         ])
vocab_size = 2994
max_seq_len = len(vocab)
padding_idx = vocab['<PAD>']

model = Transformer(height,width,n_channels,patch_size,dim,enc_head,enc_feed_forward,enc_depth,
                    dec_head,dec_feed_forward,dec_depth,max_seq_len,vocab_size,padding_idx)
#state = torch.load("captioning.pt",map_location="cpu")
#model.load_state_dict(state)
model = model.to(device)
itos = {v: k for k, v in vocab.items()}

def predict(image):
    #img_tensor = torch.from_numpy(image)
    image = Image.fromarray(image)
    img_tensor = transforms(image).unsqueeze(0).to(device)
    preds = greedy_decoding(model, img_tensor, max_seq_len,1,2)[0].detach().cpu().numpy()
    output = [itos[token] for token in preds.tolist()]
    return  " ".join(output)

if __name__ == '__main__':
    demo = gr.Interface(
        fn=predict,
        inputs=["image"],
        outputs=["text"],
        title='Image captioning'
    )
    demo.launch()
    
    

