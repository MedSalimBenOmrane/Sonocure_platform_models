import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
import torchvision.transforms as tt
from PIL import Image
import io
import base64
import torch.nn as nn
import torch.nn.functional as F
# Define the UNet model (same as your original code)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

# Initialize the Flask app
app = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(3, 1).to(device)
model.load_state_dict(torch.load('model/model.pth', map_location=device))
model.eval()


def process_and_predict(image, model, device, depth_threshold=6):
    # Convert PIL Image to NumPy array
    image = np.array(image)
    original_image = image.copy()  # Keep the original image for drawing

    # Define the target size for model input
    target_size = (128, 128)

    # Resize image for prediction
    resized_image = cv2.resize(image, target_size)
    image_tensor = torch.tensor(resized_image.astype(np.float32) / 255.).unsqueeze(0).permute(0, 3, 1, 2)
    image_tensor = tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image_tensor)

    with torch.no_grad():
        pred = model(image_tensor.to(device))
        pred = pred.detach().cpu().numpy()[0, 0, :, :]

    # Threshold prediction to create binary mask
    binary_mask = (pred > 0.5).astype(np.uint8)

    # Resize binary mask back to original image size
    binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Find contours on the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw contour lines on the original image
        cv2.drawContours(original_image, [largest_contour], -1, (0, 255, 0), 2)  # Green contour line

        # Draw bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box

        # Calculate tumor size
        # Calculate scaling factor
        scale_x = image.shape[1] / target_size[0]
        scale_y = image.shape[0] / target_size[1]
        area_pixels = cv2.contourArea(largest_contour)
        area_cm2 = area_pixels * (0.026458 ** 2) * (scale_x * scale_y)  # Adjust size conversion

        tumor_size = area_cm2

        # Calculate tumor depth
        M = cv2.moments(largest_contour)
        cX, cY = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)
        dist_to_left = cX - x
        dist_to_right = (x + w) - cX
        dist_to_top = cY - y
        dist_to_bottom = (y + h) - cY
        min_dist_to_edge = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        is_superficial = min_dist_to_edge < depth_threshold
    else:
        tumor_size, is_superficial = 0, None

    # Encode the image with contours as bytes
    _, img_encoded = cv2.imencode('.jpg', original_image)
    img_bytes = img_encoded.tobytes()
    return img_bytes, tumor_size, is_superficial


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_bytes, tumor_size, is_superficial = process_and_predict(image, model, device)

            # Save image to file
            output_path = 'output_image.jpg'
            with open(output_path, 'wb') as f:
                f.write(img_bytes)

            # Return path to saved image
            response = {
                'image_path': output_path,
                'size_cm': tumor_size,
                'depth': 'superficial' if is_superficial else 'deep'
            }
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': str(e)})

    return jsonify({'error': 'File processing failed'})


if __name__ == '__main__':
    app.run(debug=True)