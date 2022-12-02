from torchvision import models, transforms
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F

def predict(image_path, cfg):
    checkpoint = ('resnet50_models/best_validation_weights_acc.pt')

    num_class = cfg.data.num_classes
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_class)

    weights = torch.load(checkpoint, map_location=torch.device('cpu'))
    resnet.load_state_dict(weights, strict=True)

    #https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
    transforms.Resize((cfg.data.input_size)),
    transforms.CenterCrop(cfg.data.input_size),
    transforms.ToTensor(),
    transforms.Normalize(cfg.data.mean, cfg.data.std)
    ])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    resnet.eval()
    out = resnet(batch_t)

    with open('dr_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0]
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:]]
