from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models
import json
import argparse

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = model = getattr(models, checkpoint['arch'])(pretrained=True)

    classifier = nn.Sequential(nn.Linear(25088, 512),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(512, 1024),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(1024, 102),
                           nn.LogSoftmax(dim = 1)
                          )

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    #Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)

    pil_image.thumbnail((256, 256))

    width, height = pil_image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    pil_image = pil_image.crop((left, top, right, bottom))

    np_image = np.array(pil_image) / 255.0

    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    np_image = np_image.transpose((2, 0, 1))

    image_tensor = torch.tensor(np_image, dtype=torch.float)

    return image_tensor

def predict(args):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    #Implement the code to predict the class from an image file
    model = load_checkpoint('checkpoint.pth')

    with torch.no_grad():
        logps = model.forward(process_image(args.input).unsqueeze(0))
        ps = torch.exp(logps)
        probs, labels = ps.topk(args.top_k, dim=1)

        class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = list()

        for label in labels.numpy()[0]:
            classes.append(class_to_idx_inv[label])

        probs = probs.numpy()[0]

        if args.category_names:
            with open(args.category_names, 'r') as f:
                cat_to_name = json.load(f)
            classes = [cat_to_name[str(label)] for label in classes]

        for i in range(len(probs)):
            print(f"Class: {classes[i]}, Probability: {probs[i]*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help="path of input image")
    parser.add_argument('checkpoint', type=str, help="checkpoint file")
    parser.add_argument('--top_k', type=int, default=5, help="number of top classes to show")
    parser.add_argument('--category_names', type=str, help="JSON file for category names")
    parser.add_argument('--gpu', action='store_true', help="use GPU if available")
    args = parser.parse_args()
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    predict(args)