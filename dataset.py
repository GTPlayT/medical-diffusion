import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms
from typing import Optional

from diffusion_model import DiffusionModel
from interpolate import Transform
from config import Datatypes

class ChestXray(Dataset):
    def __init__(
            self, 
            diffusion_model: Optional[DiffusionModel] = None, 
            transform: Optional[Transform] = None, 
            indices: Optional[list[int]] = [0, 1, 2, 3, 4],
            datatype: Datatypes = Datatypes.FLOAT32
            ):
        super().__init__()
        self.mapping = {
            'NORMAL': 0,
            'PNEUMONIA': 1
        }
        self.annotations = pd.read_csv("labels.csv").replace({'label': self.mapping})
        self.transform = transform
        
        if diffusion_model is None:
            diffusion_model = DiffusionModel()
        
        self.feature_extractor = diffusion_model.feature_extractor
        self.tokenizer = diffusion_model.tokenizer
        self.text_encoder = diffusion_model.text_encoder

        self.datatype = datatype
        self.indices = indices


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_label = self.annotations.iloc[index, 0]
        image_path = self.annotations.iloc[index, 1]

        image_label = torch.tensor(int(image_label))
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        features = self.feature_extractor(images=image, return_tensors="pt").pixel_values[0].to(self.datatype.to_torch_dtype())
        
        image = torchvision.transforms.Normalize([0.5], [0.5])(image)

        if image_label == 0:
            label = f"normal {index}"
        else:
            label = f"pneumonia {index}"

        inputs = self.tokenizer(label, padding='max_length', max_length=8, truncation=True, return_tensors="pt")
        
        ids = inputs.input_ids[0]
        mask = inputs.attention_mask[0]
        
        hidden_states = self.text_encoder(ids.unsqueeze(0)).last_hidden_state[0]


        out_list = [image, label, features, image_label, ids, hidden_states, mask]

        return [out_list[i] for i in self.indices]
