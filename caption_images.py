from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import sys
from models.blip import blip_decoder
import json
import os

IMAGE_SIZE = 384

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_demo_image(image_path, image_size, device):
    raw_image = Image.open(str(image_path)).convert('RGB')
    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image



image_url_list=sys.argv[1]

device=get_device()
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
model = blip_decoder(pretrained=model_url, image_size=IMAGE_SIZE, vit='base')
model.eval()
model = model.to(device)

for image_url in open(image_url_list):
    image=load_demo_image(image_url.strip(),IMAGE_SIZE,device)
    with torch.no_grad():
        # beam search
        #caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        caption = model.generate(image, sample=True, top_p=0.9, max_length=40, min_length=5)
        result=json.dumps({'image_id':os.path.basename(image_url.strip()),'captions':caption})
        print(result)
                      
