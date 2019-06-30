# Helper function for extracting features from pre-trained models
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import os
from .utils import l2_norm, hflip_batch


def extract_feature(image, backbone, model_root, input_size=[112, 112], rgb_mean=[0.5, 0.5, 0.5],
                    rgb_std=[0.5, 0.5, 0.5], device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta=True):

    # define transform
    transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
        transforms.CenterCrop([input_size[0], input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std)])

    if isinstance(image, list):
        image = [transform(i).unsqueeze(0) for i in image]
    else:
        image = transform(image).unsqueeze(0)

    # load backbone from a checkpoint
    # print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)

    # extract features
    backbone.eval()  # set to evaluation mode

    with torch.no_grad():
        if isinstance(image, list):
            embedding = []
            if tta:
                for i in image:
                    fliped = hflip_batch(i)
                    embedding.append(backbone(i.to(device)).cpu() + backbone(fliped.to(device)).cpu())
            else:
                for i in image:
                    embedding.append(l2_norm(backbone(i.to(device))).cpu())
        else:
            if tta:
                fliped = hflip_batch(image)
                embedding = backbone(image.to(device)).cpu() + backbone(fliped.to(device)).cpu()
            else:
                embedding = l2_norm(backbone(image.to(device))).cpu()

    #     np.save("features.npy", features)
    #     features = np.load("features.npy")

    return embedding


def extract_feature_folder(data_root, backbone, model_root, input_size=[112, 112], rgb_mean=[0.5, 0.5, 0.5],
                    rgb_std=[0.5, 0.5, 0.5], embedding_size=512, batch_size=512,
                    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta=True):

    # define data loader
    transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
        transforms.CenterCrop([input_size[0], input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    dataset = datasets.ImageFolder(data_root, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # load backbone from a checkpoint
    print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)

    # extract features
    backbone.eval()  # set to evaluation mode
    idx = 0
    features = np.zeros([len(loader.dataset), embedding_size])
    with torch.no_grad():
        iter_loader = iter(loader)
        while idx + batch_size <= len(loader.dataset):
            batch, _ = iter_loader.next()
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = backbone(batch.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                features[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                features[idx:idx + batch_size] = l2_norm(backbone(batch.to(device))).cpu()
            idx += batch_size

        if idx < len(loader.dataset):
            batch, _ = iter_loader.next()
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = backbone(batch.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                features[idx:] = l2_norm(emb_batch)
            else:
                features[idx:] = l2_norm(backbone(batch.to(device)).cpu())
                
#     np.save("features.npy", features) 
#     features = np.load("features.npy")

    return features



