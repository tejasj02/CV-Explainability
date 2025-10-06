# AI was used in referencing apis for model and data loading
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# gpu/cpu check
def get_device():
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("⚠️ GPU not available, using CPU")
        return torch.device("cpu")

# perform GRADCAM and variants on 5 images using pretrained resnet 50
def main():
    device = get_device()

    # Image and label setup
    img_labels = ['Shirt', 'Shoes', 'Shorts', 'Pants', 'Dress']
    img_names = [
        'ea7b6656-3f84-4eb3-9099-23e623fc1018',
        '3b86d877-2b9e-4c8b-a6a2-1d87513309d0',
        '5d3a1404-697f-479f-9090-c1ecd0413d27',
        'c995c900-693d-4dd6-8995-43f3051ec488',
        'e3c8e575-c5b8-4c4c-9f49-62b37b611b6b'
    ]
    image_dir = "images_original"

    # Load pretrained model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval().to(device)
    target_layer = model.layer4[-1]

    # transform and resize
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # get class names
    from torchvision.models import ResNet50_Weights
    imagenet_labels = ResNet50_Weights.DEFAULT.meta["categories"]

    # Iterate over images
    for name, lbl in zip(img_names, img_labels):
        img_path = os.path.join(image_dir, f"{name}.jpg")
        img = Image.open(img_path).convert("RGB")
        rgb_img = np.array(img) / 255.0
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            preds = model(input_tensor)
            pred_idx = preds.argmax(dim=1).item()
            pred_label = imagenet_labels[pred_idx]
        print(f"🖼️ {lbl}: predicted as {pred_label}")

        #GRAD-CAMS
        cam_methods = {
            "Grad-CAM": GradCAM,
            "Grad-CAM++": GradCAMPlusPlus,
            "Score-CAM": ScoreCAM
        }

        plt.figure(figsize=(12,4))
        for i, (name_cam, cam_class) in enumerate(cam_methods.items()):
            cam = cam_class(model=model, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=input_tensor)[0, :]
            grayscale_cam_resized = np.array(Image.fromarray(grayscale_cam).resize((rgb_img.shape[1], rgb_img.shape[0]), Image.BILINEAR))
            cam_image = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=True)

            plt.subplot(1,3,i+1)
            plt.imshow(cam_image)
            plt.title(name_cam)
            plt.axis("off")

        plt.suptitle(f"{lbl} → Predicted: {pred_label}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
