import argparse
import os
import time
from pathlib import Path
from typing import List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from boxmot import StrongSORT
from ultralytics import YOLO

from sam2.build_sam import build_sam2_object_tracker

SAM_CHECKPOINT_FILEPATH = "./checkpoints/sam2.1_hiera_base_plus.pt"
YOLO_CHECKPOINT_FILEPATH = "yolov8x-seg.pt"
BOXMOT_CHECKPOINT_FILEPATH = "osnet_x0_25_msmt17.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_b+.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrator Argument Parser")

    parser.add_argument('--source',
                        type=str,
                        required=True,
                        help='Path to the data source'
                        )
    parser.add_argument('--num_objects',
                        type=int,
                        default=2,
                        help='Maximum number of objects to track with SAM'
                        )
    parser.add_argument('--labels',
                        type=int,
                        nargs='+',
                        default=None,
                        help='YOLO class labels to detect (e.g., 0 1 2)'
                        )
    parser.add_argument('--device',
                        type=str,
                        default='cuda:1',
                        help='Device to run the SAM and YOLO model on'
                        )
    parser.add_argument('--boxmot',
                        action="store_true",
                        help='Whether to use multi-object tracking algorithms instead of SAM'
                        )
    parser.add_argument('--visualize',
                        action="store_true",
                        help='Whether to visualize tracking'
                        )

    args = parser.parse_args()

    return args


class SAMTracker:
    def __init__(self,
                 config_file,
                 ckpt_path,
                 num_objects=10,
                 device='cuda:1',
                 iou_threshold=0.9
                 ):

        self.config_file = config_file
        self.ckpt_path = ckpt_path
        self.num_objects = num_objects
        self.device = device
        self.iou_threshold = iou_threshold
        self.sam = build_sam2_object_tracker(num_objects=self.num_objects,
                                             config_file=self.config_file,
                                             ckpt_path=self.ckpt_path,
                                             device=self.device
                                             )
        self.sam.use_mask_input_as_output_without_sam = True
        self.current_vision_feats = None
        self.current_vision_pos_embed = None
        self.feat_sizes = None
        self.image_size = self.sam.image_size

    def compute_iou_mask(self, mask1: torch.Tensor, mask2: torch.Tensor) -> float:
        """
        Compute the Intersection over Union (IoU) between two binary masks.

        Parameters
        ----------
        mask1 : torch.Tensor
            First binary mask (H, W) on the same device.

        mask2 : torch.Tensor
            Second binary mask (H, W) on the same device.

        Returns
        -------
        float
            IoU value between mask1 and mask2.

        """
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        intersection = torch.sum(mask1 & mask2).item()
        union = torch.sum(mask1 | mask2).item()

        return intersection / union if union > 0 else 0

    def get_max_overlap_per_object(self, object_masks: torch.Tensor, sam_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute the maximum IoU for each object mask across all SAM masks.

        Parameters
        ----------
        object_masks : torch.Tensor
            Binary object masks (N, H, W).

        sam_masks : torch.Tensor
            Binary SAM masks (M, H, W).

        Returns
        -------
        torch.Tensor
            A tensor of shape (N,) containing the maximum IoU for each object mask.

        """

        # Compute IoU for all combinations of object_masks and sam_masks
        intersection = (object_masks.unsqueeze(1) & sam_masks.unsqueeze(0)).float().sum(dim=(2, 3))  # (n, y)
        union = (object_masks.unsqueeze(1) | sam_masks.unsqueeze(0)).float().sum(dim=(2, 3))  # (n, y)
        iou = intersection / (union + 1e-6)  # (n, y), adding epsilon to avoid division by zero

        # Get the max IoU for each object_mask across all sam_masks
        max_iou, _ = iou.max(dim=1)

        return max_iou

    def plot_mask_overlap(self, sam_masks: torch.Tensor, object_masks: torch.Tensor) -> None:
        """
        Visualize the overlap between SAM masks and object masks.

        Parameters
        ----------
        sam_masks : torch.Tensor
            Binary SAM masks (M, H, W).

        object_masks : torch.Tensor
            Binary object masks (N, H, W).

        """
        # Initialize an empty RGB image
        height, width = object_masks.shape[1:3]
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Assign a random color to each object mask
        for i, obj_mask in enumerate(object_masks):
            color = np.random.randint(0, 255, size=3)
            image[obj_mask.cpu().numpy()] = color

        # Overlay the SAM mask in white
        for sam_mask in sam_masks:
            image[sam_mask.cpu().numpy()] = [255, 255, 255]

        # Plot the result
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def check_for_new_objects(self, object_masks: torch.Tensor, sam_masks: torch.Tensor) -> np.ndarray:
        """
        Identify new object masks not overlapping with SAM masks.

        Parameters
        ----------
        object_masks : torch.Tensor
            Binary object masks (N, H, W).

        sam_masks : torch.Tensor
            Binary SAM masks (M, H', W').

        Returns
        -------
        np.ndarray
            Array of new object masks with low overlap with SAM masks.

        """

        # Convert to boolean
        object_masks = object_masks > .5

        new_obj_indexes = set(range(len(object_masks)))

        sam_masks = torch.nn.functional.interpolate(sam_masks,
                                                    size=(object_masks[0].shape[0], object_masks[0].shape[1]),
                                                    mode='bilinear',
                                                    align_corners=False
                                                    )

        # Convert to boolean
        sam_masks = sam_masks > 0
        sam_masks = sam_masks[:, 0]

        # Plot the overlap between SAM masks and object masks
        # self.plot_mask_overlap(sam_masks, object_masks)

        # get maximum iou for each object mask across all sam masks
        iou = self.get_max_overlap_per_object(object_masks=object_masks,
                                              sam_masks=sam_masks
                                              )

        # Find indices where max IoU is below the threshold
        overlapping_indices = (iou >= self.iou_threshold).nonzero(as_tuple=True)[0]

        # Remove overlapping object indices from new_obj_indexes
        new_obj_indexes -= set(overlapping_indices.tolist())

        new_object_masks = object_masks[list(new_obj_indexes)]

        new_object_masks = new_object_masks.cpu().numpy()

        return new_object_masks


    def track_frame(self, img: np.ndarray) -> Dict:
        """
        Track existing objects.

        Parameters
        ----------
        img : np.ndarray
            Input image (H, W, C), where H is height, W is width, and C is channels.

        Returns
        -------
        prediction : Dict
            The prediction result.

        """

        print('---=====[SAM Track Existing Objects]=====---')

        start_time = time.time()
        img = self.sam.preprocess_image(img=img)
        print(f"Preprocessing: {(time.time() - start_time) * 1000:.1f} ms")

        start_time = time.time()
        img_features = self.sam.get_image_features(img=img)
        self.current_vision_feats, self.current_vision_pos_embeds, self.feat_sizes = img_features
        print(f"Image Embedding: {(time.time() - start_time) * 1000:.1f} ms")

        start_time = time.time()
        prediction = self.sam.inference(current_vision_feats=self.current_vision_feats,
                                        current_vision_pos_embeds=self.current_vision_pos_embeds,
                                        feat_sizes=self.feat_sizes,
                                        point_inputs=None,
                                        mask_inputs=None,
                                        run_mem_encoder=True,
                                        prev_sam_mask_logits=None
                                        )
        print(f"Inference: {(time.time() - start_time) * 1000:.1f} ms")
        print('---======================================---\n')

        return prediction


    def track_new_object(self, new_object_masks: np.ndarray, tracked_object_masks: np.ndarray) -> Dict:
        """
        Track new objects by processing the provided new object masks and updating SAM's memory.

        Parameters
        ----------
        new_object_masks : np.ndarray
            New binary object masks (n, height, width) where n is the number of new objects.

        tracked_object_masks : torch.Tensor
            Existing binary object masks (m, height, width) where m is the number of tracked objects.

        Returns
        -------
        prediction : Dict
            The prediction result.

        """

        print('---=====[SAM Track New Object]=====---')

        available_slots = self.num_objects - self.sam.curr_obj_idx

        start_time = time.time()
        # Assign masks to this SAM instance
        new_object_masks = new_object_masks[0: min(available_slots, new_object_masks.shape[0])]

        new_object_masks = torch.from_numpy(new_object_masks)
        new_object_masks = new_object_masks[:, None]  # Add channel dimension -> (n, 1, height, width)
        new_object_masks = torch.nn.functional.interpolate(new_object_masks.float(),
                                                           size=(self.sam.image_size, self.sam.image_size),
                                                           align_corners=False,
                                                           mode="bilinear",
                                                           antialias=True,  # use antialias for downsampling
                                                           )
        new_object_masks = new_object_masks >= 0.5

        new_object_masks = new_object_masks.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)

        # Resize prediction masks and ensure correct dtype and binary format
        mask_inputs = torch.nn.functional.interpolate(tracked_object_masks,
                                                      size=(self.sam.image_size, self.sam.image_size),
                                                      mode="bilinear",
                                                      align_corners=False,
                                                      antialias=True
                                                      )
        mask_inputs = (mask_inputs > 0).to(dtype=torch.bfloat16, non_blocking=True)

        # Update the correct slice with mask inputs
        mask_inputs[self.sam.curr_obj_idx:self.sam.curr_obj_idx + new_object_masks.shape[0]] = new_object_masks
        self.sam.curr_obj_idx += new_object_masks.shape[0]
        print(f"Mask Inputs: {(time.time() - start_time) * 1000:.1f} ms")

        start_time = time.time()
        prediction = self.sam.inference(current_vision_feats=self.current_vision_feats,
                                        current_vision_pos_embeds=self.current_vision_pos_embeds,
                                        feat_sizes=self.feat_sizes,
                                        point_inputs=None,
                                        mask_inputs=mask_inputs,
                                        run_mem_encoder=True,
                                        prev_sam_mask_logits=None
                                        )
        print(f"Inference: {(time.time() - start_time) * 1000:.1f} ms")
        print('---================================---')

        return prediction

    def update_memory_bank(self, prediction):
        """
        Update the memory bank of the SAM model with the latest prediction.

        Parameters
        ----------
        prediction : dict
            The latest prediction result from the SAM model.

        """

        self.sam.update_memory_bank(prediction=prediction)


class BOXMotTracker:
    def __init__(self,
                 ckpt_path,
                 device
                 ):

        # Load tracker
        self.boxmot = StrongSORT(model_weights=Path(ckpt_path), device=device, fp16=True)

    def get_tracks(self, detections: np.ndarray, img: np.ndarray):
        print('---=====[BoxMot Track Detected Objects]=====---')

        start_time = time.time()
        tracks = self.boxmot.update(detections, img)
        print(f"Track: {(time.time() - start_time) * 1000:.1f} ms")

        print('---=========================================---\n')

        return tracks


class YOLODetector:
    def __init__(self,
                 ckpt_path,
                 conf_threshold,
                 device,
                 labels: List[int] = None
                 ):

        self.yolo = YOLO(ckpt_path)
        self.yolo.to(device)
        self.conf_threshold = conf_threshold
        self.labels = labels

    def get_detections(self, img: np.ndarray):
        print('---=====[YOLO Detect Objects]=====---')

        start_time = time.time()
        # Get YOLO predictions
        pred = self.yolo(img, conf=self.conf_threshold, verbose=False)

        if pred[0].masks is None:
            return None

        masks = pred[0].masks.data
        cls = pred[0].boxes.cls.data
        conf = pred[0].boxes.conf.data
        boxes = pred[0].boxes.xyxy

        if self.labels is not None:
            object_filter = np.where(np.isin(cls.cpu().numpy(), self.labels))
            masks = masks[object_filter]
            cls = cls[object_filter]
            conf = conf[object_filter]
            boxes = boxes[object_filter]

        if masks.shape[0] == 0:
            return None

        # Resize YOLO segmentation mask to match input image dimensions
        masks = F.interpolate(masks.unsqueeze(0),
                              size=(img.shape[0], img.shape[1]),
                              mode='bilinear',
                              align_corners=False
                              )

        # Remove first dimension to get (masks, height, width)
        masks = masks[0]
        print(f"Inference: {(time.time() - start_time) * 1000:.1f} ms")
        print('---===============================---\n')

        return {'masks': masks, 'boxes': boxes, 'cls': cls, 'conf': conf}


class Tracker:
    def __init__(self,
                 device: str,
                 labels: List = None,
                 visualize: bool = False,
                 num_objects: int = None,
                 boxmot: bool = False
                 ):

        self.visualizer = None
        self.available_slots = num_objects
        self.tracking_started = False
        self.num_objects = num_objects
        self.device = device
        self.boxmot = boxmot

        self.detector = YOLODetector(ckpt_path=YOLO_CHECKPOINT_FILEPATH,
                                     conf_threshold=0.6,
                                     device=device,
                                     labels=labels
                                     )

        if self.boxmot:
            self.tracker = BOXMotTracker(ckpt_path=BOXMOT_CHECKPOINT_FILEPATH,
                                         device=device
                                         )

        else:
            self.tracker = SAMTracker(config_file=SAM_CONFIG_FILEPATH,
                                      ckpt_path=SAM_CHECKPOINT_FILEPATH,
                                      num_objects=num_objects,
                                      device=device,
                                      iou_threshold=0.7
                                      )

        if visualize:
            from visualizer import Visualizer
            self.visualizer = Visualizer(video_width=1024,
                                         video_height=1024,
                                         num_masks=num_objects if not boxmot else 50,
                                         save_video=True,
                                         output_path='./segmented_video.mp4'
                                         )

    def get_next_frame(self, source):
        if os.path.isdir(source):
            img_filenames = [f for f in os.listdir(source) if f.endswith('.jpg')]
            img_filenames.sort()
            img_filepaths = [os.path.join(source, f) for f in img_filenames]

            for img_filepath in img_filepaths:
                img = cv2.imread(filename=img_filepath)

                yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        else:  # Assume video file path
            cap = cv2.VideoCapture(source)
            while cap.isOpened():
                ret, img = cap.read()
                if not ret:
                    break
                yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cap.release()


    def process_frame(self, frame: np.ndarray):
        # Get detections from YOLO
        detections = self.detector.get_detections(img=frame) if self.available_slots > 0 else None

        if not self.tracking_started:
            if detections is None:
                return None, None
            else:
                self.tracking_started = True

        if isinstance(self.tracker, SAMTracker):
            sam_out = self.tracker.track_frame(img=frame)

            if detections is not None:
                new_object_masks = self.tracker.check_for_new_objects(object_masks=detections['masks'],
                                                                      sam_masks=sam_out['pred_masks']
                                                                      )

                self.available_slots = self.tracker.num_objects - self.tracker.sam.curr_obj_idx

                if len(new_object_masks) > 0 and self.available_slots > 0:
                    sam_out = self.tracker.track_new_object(new_object_masks=new_object_masks,
                                                            tracked_object_masks=sam_out['pred_masks']
                                                            )

            self.tracker.update_memory_bank(prediction=sam_out)

            masks, track_ids = sam_out['pred_masks'], list(range(self.num_objects))

            return masks, track_ids

        else:
            if detections is None:
                return None, None

            dets = torch.cat((detections['boxes'],
                              detections['conf'].reshape((-1, 1)),
                              detections['cls'].reshape((-1, 1))
                              ), dim=1
                             )

            dets = dets.cpu().numpy()

            tracks = self.tracker.get_tracks(detections=dets, img=frame)

            if tracks.size == 0:
                return None, None

            inds, track_ids = tracks[:, 7].astype('int'), tracks[:, 4].astype('int')
            masks = detections['masks'][inds]

            return masks, track_ids

    def process_frames(self, source: str):
        run_time_start = time.time()

        with torch.inference_mode(), torch.autocast(self.device.split(':')[0],
                                                    dtype=torch.bfloat16 if not self.boxmot else torch.float16
                                                    ):
            for img in self.get_next_frame(source):
                masks, track_ids = self.process_frame(frame=img)

                if masks is None:
                    continue
                    
                # Visualize if needed
                if self.visualizer is not None:
                    masks = masks.unsqueeze(1) - 0.5 if self.boxmot else masks
                    self.visualizer.add_frame(frame=img, mask=masks, object_id=track_ids)


        print('Runtime:', time.time() - run_time_start)
        if self.visualizer is not None:
            self.visualizer.stop()


if __name__ == '__main__':
    args = parse_args()

    tracker = Tracker(device=args.device,
                      labels=args.labels,
                      num_objects=args.num_objects,
                      visualize=args.visualize,
                      boxmot=args.boxmot
                      )

    tracker.process_frames(source=args.source)
