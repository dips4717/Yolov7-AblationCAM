
"""
Ablationcam for YOLOv7. 
Adapted from the following https://jacobgil.github.io/pytorch-gradcam-book/

author: dipu.manandhar
"""


#%%
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, FullGrad
from utils.ablation_cam_yolo import AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN, AblationLayer, AblationLayerYOLOv7
from torchvision.models import resnet50
import torchvision.transforms as tf
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from models.yolo import Model
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, xyxy2tlwh, strip_optimizer, set_logging, increment_path

from pathlib import Path
import yaml
from models.experimental import attempt_load
import cv2
import argparse
import torchvision
import random

COLORS = np.random.uniform(0, 255, size=(9, 3))
classes = ['M1', 'M3', 'N3', 'T', 'N1', 'trailer', 'L1', 'N2', 'M2']
classid2name = {k:v for k,v in enumerate(classes)}
#weight = 'runs/train/yolov7tiny-6classPlusUKBatch2-norect416/weights/best.pt'
weight = 'runs/train/yolov7tiny-6cUKB2LithB1PlusBackground-norect416/weights/epoch_024.pt'

# classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#          'hair drier', 'toothbrush' ]
# classid2name = {k:v for k,v in enumerate(classes)}
# COLORS = np.random.uniform(0, 255, size=(80, 3))
# weight = '/data/objects/yolov7/yolov7-tiny.pt'



def txt2list(fn):
    with open(fn, 'r') as f:
        files = f.readlines()
    files = [x.strip() for x in files]
    return files 
 

def parse_predictions(preds):
    # Similar thing for our YOLO
    preds = preds[0].cpu()
       
    boxes, colors, names = [], [], []
    for pred in preds:
        if pred[4] < 0.1:
            continue
         
        xmin = int(pred[0])
        ymin = int(pred[1])
        xmax = int(pred[2])
        ymax = int(pred[3])
        category = int(pred[5])
        color = COLORS[category]
        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(classid2name[category])
    
    boxes = np.int32(boxes)
    return boxes, colors, names
    

def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color, 
            2)

        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img


##%%
def fasterrcnn_reshape_transform(x):
    target_size = x['pool'].size()[-2 : ]
    activations = []
    for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations


def yolo_reshape_transform_singlelayer(x):
    #print(len(x))
    print (f'ACTIVATION SHAPE: {x.shape}')   
    #activations = torch.abs(x)
    activations = x
    return activations



def yolo_reshape_transform2(x):
    """
    torch.Size([1, 128, 52, 52])
    torch.Size([1, 256, 26, 26])
    torch.Size([1, 512, 13, 13])
    output: tensor of shape [1, 896, 13, 13]
    """
    
    target_size = x[14].size()[-2 : ] 
    #target_size = x[-1].size()[-2 : ]
    sel_x = [x[14], x[21], x[28] ] 
    activations = []
    for value in sel_x:
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations


def yolo_reshape_transform(x):
    """
    torch.Size([1, 128, 52, 52])
    torch.Size([1, 256, 26, 26])
    torch.Size([1, 512, 13, 13])
    output: tensor of shape [1, 896, 13, 13]
    """
    
    target_size = x[0].size()[-2 : ] 
    #target_size = x[-1].size()[-2 : ] 
    activations = []
    for value in x:
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations
   
def yolo_reshape_transform_last(x):
    """
    pred[0] = before NMS out
    pred[1]: is tuple with 3 tensors with following shapes

    torch.Size([1, 3, 40, 52, 14])
    torch.Size([1, 3, 20, 26, 14])
    torch.Size([1, 3, 10, 13, 14])
    """

    print(len(x))
    acts = x[1]
    target_size = acts[0].size()[-3 : ]
    activations = []
    for value in acts:
        activations.append(torch.nn.functional.interpolate(torch.abs(value.squeeze()), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)    
    activations = torch.abs(x)
    activations = activations.unsqueeze(0)
    return activations



class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
		assign a score on how the current bounding boxes match it,
			1. In IOU
			2. In the classification score.
		If there is not a large enough overlap, or the category changed,
		assign a score of 0.

		The total score is the sum of all the box scores.
	"""
    

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    
    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        # if len(model_outputs["boxes"]) == 0:
        #      return output
        #model_outputs = model_outputs[0]
        model_output_nms = non_max_suppression(model_outputs, conf_thres=0.1, iou_thres=0.45, classes=None, agnostic=False, multi_label=True)
        model_output_nms = model_output_nms[0]
        #print(model_output_nms)
        model_output_boxes = model_output_nms[:,:4]
        model_output_labels = [classid2name [int(x.cpu())] for x in  model_output_nms[:,5] ]
        model_output_scores = model_output_nms[:,4]

        # print('New Score for the Target', model_output_boxes.shape)
        # print('Labels', model_output_labels)
        # print('Scores', model_output_scores)


        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            
            # ious = torchvision.ops.box_iou(box, model_output_boxes)
            # index = ious.argmax()
            # if ious[0, index] > self.iou_threshold and model_output_labels[index] == label:
            
            sel_index = [ii for ii, x in enumerate(model_output_labels) if x ==label]
            model_output_boxes_tmp = model_output_boxes[sel_index,:4]
            model_output_labels_tmp = [model_output_labels[x] for x in sel_index ]
            model_output_scores_tmp =  model_output_scores[sel_index]
            ious = torchvision.ops.box_iou(box, model_output_boxes_tmp)
            if ious.shape[1] >0:
                index = ious.argmax()
                # if ious[0, index] > self.iou_threshold and model_output_labels[index] == label:
                if ious[0, index] > self.iou_threshold:
                    #score = ious[0, index] + model_output_scores[index]
                    score = model_output_scores_tmp[index]
                    #print('Model outputs:  ', model_outputs.shape)
                    output = output + score
        return output


def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes


class YOLOBackbone(nn.Module):  # For yolov7-tiny.
    def __init__(self,net):
        super(YOLOBackbone, self).__init__()
        self.backbone = net.model[:77]
        self.save = net.save  
    
    def forward(self, x):
        y, dt = [], []  # outputs
        for m in self.backbone:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return y[-3:]


class YOLOBackbone2(nn.Module):  # For yolov7-tiny.
    def __init__(self,net):
        super(YOLOBackbone2, self).__init__()
        self.backbone = net.model[:29]
        self.save = net.save  
    
    def forward(self, x):
        y, dt = [], []  # outputs
        for m in self.backbone:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return y
    

class YOLOHead(nn.Module):  # For yolov7-tiny.
    def __init__(self,net):
        super(YOLOHead, self).__init__()
        self.head = net.model[29:77]
        self.save = net.save
        old_f = [14,21,28] + list (range(29,77))
        new_f = list(range(51))
        self.convert_f = {k:v for k,v in zip(old_f, new_f)}


    
    def forward(self, x):
        y, dt = x, []  # outputs
        x=y[-1]
        for m in self.head:
            #print(m.f, len(y))
            if m.f != -1:  # if not from previous layer
                if  isinstance(m.f, int):
                    if m.f < 1:
                        x = y[m.f]
                    elif m.f >0:
                        x = y[self.convert_f[m.f]] 
                else:
                     temp_x = [] 
                     for j in m.f:
                         if j == -1:
                             temp_x.append(x)
                         elif j <-1:
                             temp_x.append(y[j])
                         elif j >1:
                             temp_x.append(y[self.convert_f[j]])
                     x = temp_x

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return y[-3:]
    
    # def forward (self, x ): # x is list of actications from layer 14 21 28
    #     #activations = x.clone
    #     y, dt = x, []  # outputs
    #     x=y[-1]
    #     for m in self.head:
    #         print(m.f, len(y))
    #         if m.f != -1:  # if not from previous layer
    #             x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
    #         x = m(x)  # run
    #         y.append(x if m.i in self.save else None)  # save output
    #     return y[-3:]


class YOLOTinyRedefined(nn.Module):
    def __init__(self, original_model) -> None:
        super(YOLOTinyRedefined,self).__init__()
        self.backbone = YOLOBackbone(original_model)
        self.Idetect = original_model.model[-1]
        
    def forward(self,x):
        x = self.backbone(x)
        x = self.Idetect(x)
        return x 

class YOLOTinyRedefined2(nn.Module):
    def __init__(self, original_model) -> None:
        super(YOLOTinyRedefined2,self).__init__()
        self.backbone = YOLOBackbone2(original_model)
        self.head = YOLOHead(original_model)
        self.Idetect = original_model.model[-1]
        
    def forward(self,x):
        x = self.backbone(x)
        if len(x) == 3 : # for the call from ablation
            pass 
        else:            # for normal pass where the output is activations from all layers.
            x =  [aa for ii,aa in enumerate(x) if ii in [14,21,28]]
        x = self.head(x)
        x = self.Idetect(x)
        return x     




#%%
# img_path = '/data/objects/dataset_images/lithuania_data/lithuania_quick/images/ovr_20230321152600_fou477.jpg'
#img_path = '/data/objects/dataset_images/lithuania_data/lithuania_day/18_05_23_morning/images/2023-05-18T06_16_24.080.jpg'
#img_path = '/data/objects/dataset_images/UK_Front6Class/A33_ovr_batch1/Day/N1/ovr_20171028073148_aj02dbu.jpg'
#img_path = 'temp_images/ovr_20221211130002_yk71xrm.jpg'
#img_path = 'temp_images/ovr_20221211130005_kp70kcg.jpg'
#img_path = 'temp_images/ovr_20230212080129_1xnx99.jpg'
#img_path = 'temp_images/ovr_20230212080202_v430pd.jpg'
#img_path = 'puppies.jpg'
#img_path = '/data/objects/yolov7/temp_images/000000499018.jpg'

#img_path = '/data/objects/dataset_images/netherlands_trainval/01-05_3_23/val/ovr_20230301081723_41bgg5.jpg'
# img_path = '/data/objects/dataset_images/gradcam_testimages/cdv.JPG'
#img_path = '/data/objects/dataset_images/gradcam_testimages/cdv_3.jpg'
# img_path/ = '/data/objects/dataset_images/gradcam_testimages/ovr_20230214071037_dm690b.jpg'
# img_path = '/data/objects/dataset_images/UK_Front6Class/A33_ovr_batch1/Day/N2_processed/data/ovr_20171115140155_dg17kvt.jpg'
#img_path = '/data/objects/dataset_images/lithuania_data/lithuania_night/1000_19-22_05_23_night/images/2023-05-20T22:45:01.520.jpg'
img_path = '/data/objects/dataset_images/gradcam_testimages/cdv_4.JPG'

device = 'cuda:0'
model = attempt_load(weight, map_location=device)  
# model2 = YOLOTinyRedefined(model)
model2 = YOLOTinyRedefined2(model)
del model


do_save = False
save_path = '/data/objects/yolov7/gradcam_outputs/yolo_ablationcam/'
os.makedirs(save_path, exist_ok=True)
img_paths =[img_path]
#img_paths = txt2list('/data/objects/dataset_caches/6cUKB2LithB1val.txt')
random.shuffle(img_paths)

for ii, img_path in enumerate(img_paths):
    if ii>200:
        break
    # Dataloading/ image reading 
    img = Image.open(img_path)
    img = img.resize((416,416))
    rgb_img = img.copy() # RGB 

    # Conver PIL image to opencv 
    open_cv_image = np.array(rgb_img) 
    #open_cv_image = open_cv_image[:, :, ::-1].copy() # RGB -> BGR 
    rgb_img.show()

    img= np.float32(img)/255
    transform = tf.ToTensor() # Verify here what is this transform doing. Docs: conver Pil image or numpy (HxWxC) in range
    tensor = transform(img).unsqueeze(0).to(device) 
    #aa = tensor.cpu().numpy()

    preds2 = model2(tensor)[0]
    preds2 = non_max_suppression(preds2, conf_thres=0.1, iou_thres=0.45, classes=None, agnostic=False, multi_label=True)
    print(preds2)


    boxes, colors, names = parse_predictions(preds2)
    print('Names: ', names)
    detections = draw_detections(boxes, colors, names, open_cv_image.copy())
    Image.fromarray(detections)
   
    # CAM processing start here:
   
    target_layers = [model2.backbone]
    #targets = [FasterRCNNBoxScoreTarget(labels=names, bounding_boxes=boxes)]
    targets = [FasterRCNNBoxScoreTarget(labels=[names[4]], bounding_boxes=boxes[4:5,:])] 

    cam = AblationCAM(model2,
                    target_layers, 
                    use_cuda=torch.cuda.is_available(), 
                    reshape_transform=yolo_reshape_transform2,
                    ablation_layer=AblationLayerYOLOv7(),
                    ratio_channels_to_ablate=1.0)

    grayscale_cam = cam(tensor, targets=targets)
    # Take the first image in the batch:
    #print('GrayScaleCam Shape: ', grayscale_cam.shape)
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(open_cv_image/255, grayscale_cam, use_rgb=True)

    # And lets draw the boxes again:
    Image.fromarray(cam_image)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, cam_image)
    imfn = img_path.split('/')[-1].split('.')[0]
    save_fn = f'{save_path}/{imfn}.png'
    if do_save:
        Image.fromarray(image_with_bounding_boxes).save(save_fn)
    else:
        Image.fromarray(image_with_bounding_boxes).show()


#%%
# Re-Normalization 
# renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img, grayscale_cam)
# Image.fromarray(renormalized_cam_image)

#python gradcam.py --data data/6classes.yaml --hyp data/hyp.scratch.tiny.yaml --device 0 --weights runs/train/yolov7tiny-6classBatch1/weights/best.pt

