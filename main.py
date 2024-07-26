import pandas as pd
import torch
from tqdm import tqdm

from diffusion_model import DiffusionModel
from interpolate import Transform
from dataset import ChestXray
from classifier import Classifier

def main (
        device: str = "cpu",
        interpolation_mode: str = 'bicubic',
        image_size: int = 512,
        diffusion_model: str = 'runwayml/stable-diffusion-v1-5',
        hf_token: str = None,
        dtype: torch.dtype = torch.float32
):
    transform = Transform(interpolation_mode)
    latent_size = image_size // 8

    model = DiffusionModel(diffusion_model, hf_token, device=device)
    torch.backends.cudnn.benchmark = True

    dataset = ChestXray(model, transform=transform, indices=[0, 3])
    prompts = pd.read_csv('prompts.csv')

    inputs = model.tokenizer(
        prompts.prompt.tolist(),
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(inputs.input_ids), 100):
            embedding = model.text_encoder (
                inputs.input_ids[i:i+100].to(device)
            )[0]
            embeddings.append(embedding)
    embedding = torch.cat(embeddings, dim=0)

    ids_range = range(len(dataset))

    n = len(dataset) - 1
    digits = 0
    while n > 0:
        digits += 1
        n //= 10    
    formated_string = f"{{:0{digits}d}}"

    correct = 0
    total = 0
    progress_bar = tqdm(list(ids_range))

    Classifier.unet = model.unet
    Classifier.scheduler = model.scheduler
    Classifier.latent_size = latent_size

    for i in progress_bar:
        image, label = dataset[i]
        with torch.no_grad():
            image = image.to(device).unsqueeze(0)
            if dtype == torch.float16:
                image = image.half()
            
            x = model.vae.encode(image).latent_dist.mean
            x *= 0.18215 
        
        pred_id, _ = Classifier(x, embedding)
        pred = prompts.classidx[pred_id]

        if pred == label:
            correct += 1

        total += 1

if __name__ == "__main__":

    main()