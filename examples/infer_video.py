import time
import cv2
import numpy as np
import torch
from IPython.display import clear_output, display
from PIL import Image
from sam2.build_sam import build_sam2_object_tracker

import gc
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# List to store the clicked points
objects = []
points = []

# Mouse callback function
def register_bb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append([x, y])
            print(f"Point {len(points)}: ({x}, {y})")
            # Optional: draw a circle on the clicked point
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("First Frame", frame)

class Visualizer:
    def __init__(self,
                 video_width,
                 video_height,
                 fps=30, output_path="output.mp4"
                 ):

        self.video_width = video_width
        self.video_height = video_height
        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))

    def resize_mask(self, mask):
        mask = torch.tensor(mask, device='cpu')
        # mask = mask.detach().clone()
        mask = torch.nn.functional.interpolate(mask,
                                               size=(self.video_height, self.video_width),
                                               mode="bilinear",
                                               align_corners=False,
                                               )

        return mask

    def draw_seg_mask(self, frame, mask, current_fps):
        frame = frame.copy()
        frame = cv2.resize(frame, (self.video_width, self.video_height))
        mask = self.resize_mask(mask=mask)
        mask = (mask > 0.0).numpy()
        for i in range(mask.shape[0]):
            obj_mask = mask[i, 0, :, :]
            frame[obj_mask] = [255, 105, 180]
        # Overlay FPS text
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def add_frame(self, frame, mask, current_fps):
        processed_frame = self.draw_seg_mask(frame, mask, current_fps)
        self.video_writer.write(processed_frame)

    def release(self):
        # Crucial: This closes the file and saves the video
        self.video_writer.release()
        print("Video saved successfully.")

# Set SAM2 Configuration
NUM_OBJECTS = 1
SAM_CHECKPOINT_FILEPATH = "../checkpoints/sam2.1_hiera_tiny.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_t.yaml"
DEVICE = 'cuda:0'

clear_memory()
sam = build_sam2_object_tracker(num_objects=NUM_OBJECTS,
                                config_file=SAM_CONFIG_FILEPATH,
                                ckpt_path=SAM_CHECKPOINT_FILEPATH,
                                device=DEVICE,
                                verbose=True
                                )

video_stream = cv2.VideoCapture('../assets/dirt_bike.mp4')

video_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))

# For real-time visualization
visualizer = Visualizer(video_width=video_width,
                        video_height=video_height
                        )

available_slots = np.inf

first_frame = True
with torch.inference_mode(), torch.autocast('cuda:0', dtype=torch.bfloat16):
    while video_stream.isOpened():
        start_time = time.time()

        # Get next frame
        ret, frame = video_stream.read()

        # Exit if no frames remaining
        if not ret:
            break

        # Convert frame from BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Simulate detection on first frame
        if first_frame:
            bbox = np.array([[[487, 371], [677, 521]]]
                            )

            sam_out = sam.track_new_object(img=img,
                                           box=bbox
                                           )

            first_frame = False

        else:
            sam_out = sam.track_all_objects(img=img)

        end_time = time.time()
        fps = 1.0 / (end_time - start_time)

        visualizer.add_frame(frame=frame, mask=sam_out['pred_masks'], current_fps=fps)

clear_memory()
visualizer.release()
video_stream.release()