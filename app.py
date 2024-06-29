from flask import Flask, Response, request, render_template
from torchvision import transforms
from PIL import Image
from waitress import serve
import io
from torchvision.models import mobilenet_v2
import torch
import json

app = Flask(__name__)

class_index = json.load(open('classes_dict.json'))
nn_model = mobilenet_v2(num_classes=1382)
nn_model.load_state_dict(torch.load('mushroom_id.pt'))
# Apply softmax activation to get a probability / "confidence"
nn_model.classifier = torch.nn.Sequential(
    nn_model.classifier,
    torch.nn.Softmax(dim=1)
)
nn_model.eval()

# Resize image for model, convert to tensor, and normalize
def transform_image(image_bytes):
    img_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return img_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = nn_model.forward(tensor)
    confidence, y_hat = torch.topk(outputs, 10, 1)
    confidence = [round(100*conf.item(), 1) for conf in confidence.squeeze(0)]
    print(confidence)
    pred_str = [class_index[predicted_idx.item()].
                replace("_", " ").title().split(' ', 1)[1] for predicted_idx in y_hat.squeeze(0)]
    # pred_str = pred_str.replace("_", " ")
    return pred_str, confidence

# API call for predicting a mushroom from an input image
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files["mushroom"]
            img_bytes = file.read()
            class_name, confidence = get_prediction(image_bytes=img_bytes)
        except Exception as e:
            return Response(
                "Error processing image: " + str(e),
                status=400,
            )

        return render_template("predict.html",
                               pred_1=class_name[0],  # Assuming class_name is a list or tuple
                               conf_1=confidence[0],
                               pred_2=class_name[1],  # Assuming class_name is a list or tuple
                               conf_2=confidence[1],
                               pred_3=class_name[2],  # Assuming class_name is a list or tuple
                               conf_3=confidence[2],
                               pred_4=class_name[3],  # Assuming class_name is a list or tuple
                               conf_4=confidence[3],
                               pred_5=class_name[4],  # Assuming class_name is a list or tuple
                               conf_5=confidence[4])  # Assuming confidence is a list or tuple

    
@app.route('/')
def index():
    return render_template('index.html')
    
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)