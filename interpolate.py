import torch
from PIL import Image
import torchvision.transforms
from torchvision.transforms.functional import InterpolationMode

class Transform (torchvision.transforms.Compose):
    def __init__(self, interpolation_mode = 'bicubic', size=512):
        valid_modes = {
            'nearest': InterpolationMode.NEAREST,
            'nearest_exact': InterpolationMode.NEAREST_EXACT,
            'bilinear': InterpolationMode.BILINEAR,
            'bicubic': InterpolationMode.BICUBIC,
            'box': InterpolationMode.BOX,
            'hamming': InterpolationMode.HAMMING,
            'lancoz': InterpolationMode.LANCZOS
        }

        if interpolation_mode not in valid_modes:
                    raise ValueError(f"Invalid interpolation mode: {interpolation_mode}")

        self.interpolation_mode = valid_modes[interpolation_mode]

        self.transforms = [
            torchvision.transforms.Resize(size, interpolation=valid_modes[interpolation_mode]),
            torchvision.transforms.CenterCrop(size),
            self._convert_image_to_rgb,
            torchvision.transforms.ToTensor(),
        ]

        super().__init__(self.transforms)

    def _convert_image_to_rgb (self, image):
           return image.convert("RGB")
    
def test ():
    transform = Transform()
    image = torch.randn((1, 3, 256, 256))
    image = image.squeeze().permute(1, 2, 0)
    image = image.to(torch.uint8).cpu().numpy()
    transform(Image.fromarray(image))
    print("The transform code works.")

if __name__ == "__main__":
      test()