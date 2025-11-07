import torch
import cv2
import numpy as np
from imutils.video import VideoStream
from datetime import datetime
import time
from midas.model_loader import default_models, load_model
import utils

@torch.no_grad()
def process(device, model, image, input_size, target_size):
    sample = torch.from_numpy(image).to(device).unsqueeze(0)
    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        .squeeze()
        .cpu()
        .numpy()
    )
    return prediction

def create_side_by_side(image, depth):
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max-depth_min)
    normalized_depth *= 3
    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)      
    return np.concatenate((image, right_side), axis=1)

def run():
    time_start = time.time()
    device = torch.device("cpu")
    optimize = False
    side = False
    height = None
    square = False
    grayscale = False
    model_type = "midas_v21_small_256"
    model_path = default_models[model_type]
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)
    fps = 1
    video = VideoStream(0).start()
    while True:
        frame = video.read()
        if frame is not None:
            frame = cv2.pyrDown(frame)
            original_image_rgb = np.flip(frame, 2)
            image = transform({"image": original_image_rgb / 255})["image"]
            prediction = process(device, model, image, (net_w, net_h), original_image_rgb.shape[1::-1])
            original_image_bgr = np.flip(original_image_rgb, 2)
            content = create_side_by_side(original_image_bgr, prediction)
            cv2.imshow("MiDaS", content / 255)
            alpha = 0.1
            print(f"\rFPS: {(1 - alpha) * fps + alpha * 1 / (time.time()-time_start)}")
                
            if cv2.waitKey(1) == 27:
                break
    
    cv2.destroyAllWindows()
    video.stop()
if __name__ == "__main__":
    run()