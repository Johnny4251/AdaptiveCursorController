import cv2
import torch
import pyautogui
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import torchvision.transforms.functional as F


SCALING_FACTOR = 60 # Sensitivity
RIGHT_EYE_INDEX = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = keypointrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()
cap = cv2.VideoCapture(0)

prev_position = None


def draw_keypoints(outputs, frame, prev_position, alpha=0.2):
    ema_dx = 0
    ema_dy = 0
    scores = outputs[0]['scores'].cpu().numpy()
    keypoints = outputs[0]['keypoints'].cpu().numpy()

    high_confidence_indices = scores > 0.90
    keypoints = keypoints[high_confidence_indices]

    for keypoint in keypoints:
        current_position = (int(keypoint[RIGHT_EYE_INDEX, 0]), int(keypoint[RIGHT_EYE_INDEX, 1]))
        if prev_position is not None:
            dx = (current_position[0] - prev_position[0]) * SCALING_FACTOR
            dy = (current_position[1] - prev_position[1]) * (SCALING_FACTOR * 2)
            ema_dx = alpha * dx + (1 - alpha) * ema_dx
            ema_dy = alpha * dy + (1 - alpha) * ema_dy
            new_x = pyautogui.position()[0] + ema_dx
            new_y = pyautogui.position()[1] + ema_dy
            pyautogui.moveTo(int(new_x), int(new_y))
        prev_position = current_position
        cv2.circle(frame, current_position, 5, (0, 255, 0), thickness=-1)
    return prev_position


# Main Loop
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad(): output = model(frame_tensor)

    prev_position = draw_keypoints(output, frame, prev_position)

    cv2.imshow('Debug Window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
