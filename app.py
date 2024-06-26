from flask import Flask, jsonify, request, Response
from torchvision import transforms
from PIL import Image
import io
from torchvision.models import mobilenet_v2
import torch
import json

app = Flask(__name__)

class_index = json.load(open('classes_dict.json'))
nn_model = mobilenet_v2(num_classes=215)
nn_model.load_state_dict(torch.load('mushroom_identifier.pt'))
# Apply softmax activation to get a probability / "confidence"
nn_model.classifier = torch.nn.Sequential(
    nn_model.classifier,
    torch.nn.Softmax(dim=1)
)
nn_model.eval()

# Resize image for model, convert to tensor, and normalize
def transform_image(image_bytes):
    img_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return img_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = nn_model.forward(tensor)
    confidence, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    pred_str = class_index[predicted_idx]
    pred_str = pred_str.replace("_", " ")
    return pred_str.title(), confidence.item()

with open("chicken_of_the_woods_example.jpg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))

# API call for predicting a mushroom from an input image
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = -1
        confidence = -1
        # If get_prediction fails, return an error message
        try:
            class_name, confidence = get_prediction(image_bytes=img_bytes)
        except:
            return  Response(
                    "Unable to read the given image",
                    status=400,
                )
        return jsonify({'class_name': class_name, 'confidence': confidence})