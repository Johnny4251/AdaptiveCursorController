import cv2
import torch
import pyautogui
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import torchvision.transforms.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = keypointrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

cap = cv2.VideoCapture(0)

prev_position = None

# Used to increase sensitivity
SCALING_FACTOR = 15

def draw_keypoints(outputs, frame, prev_position):
    keypoints_indices = [2] 

    for i in range(len(outputs[0]['keypoints'])):
        keypoints = outputs[0]['keypoints'][i].cpu().numpy()
        scores = outputs[0]['scores'][i].cpu().numpy()
        if scores > 0.90:
            for idx in keypoints_indices:
                keypoint = keypoints[idx]
                current_position = (int(keypoint[0]), int(keypoint[1]))
                if prev_position is not None:
                    dx = (current_position[0] - prev_position[0]) * SCALING_FACTOR
                    dy = (current_position[1] - prev_position[1]) * (SCALING_FACTOR*2) # makes upwards head movement more sensitive
                    new_x = pyautogui.position()[0] + dx
                    new_y = pyautogui.position()[1] + dy
                    pyautogui.moveTo(new_x, new_y)
                prev_position = current_position
                cv2.circle(frame, current_position, 5, (0, 255, 0), thickness=-1)
    return prev_position

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad(): output = model(frame_tensor)

    prev_position = draw_keypoints(output, frame, prev_position)

    cv2.imshow('Debug Window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
