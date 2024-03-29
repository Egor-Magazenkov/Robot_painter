"""Using Mask-RCNN code to crop main objects from background."""
import os
import sys
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize
import cv2
import imutils

class MaskRCNNConfig(Config):
    """Config for Mask-RCNN model."""
    NAME = "coco_pretrained_model_config"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85 # минимальный процент отображения прямоугольника
    NUM_CLASSES = 81

if len(sys.argv) != 3:
    print('Bad argument usage.\n\
            Usage: python3 <path/to/image> <path/to/dataset>')
    sys.exit(1)
else:
    image = cv2.imread(sys.argv[1])
    COCO_MODEL_PATH = sys.argv[2]


model = modellib.MaskRCNN(mode="inference", model_dir=os.getcwd(), config=MaskRCNNConfig())
model.load_weights(COCO_MODEL_PATH, by_name=True)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=512)

r = model.detect([image], verbose=1)[0]
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle',
               'bus', 'train', 'truck', 'boat',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            CLASS_NAMES, r['scores'])
for i in range(len(r['rois'])):
    mask = image.copy()
    mask[r['masks'][:,:,i] == 0] = (255,255,255)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    cv2.imshow(f'mask{i}', mask)
    cv2.waitKey(0)
    cv2.imwrite(f'mask{i}.png', mask)
    mask = image.copy()
    mask[r['masks'][:,:,i] == 1] = (255,255,255)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    cv2.imshow(f'umask{i}', mask)
    cv2.waitKey(0)
    cv2.imwrite(f'umask{i}.png', mask)
