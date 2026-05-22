import os
import cv2
import random
import numpy as np
import xml.etree.ElementTree as ET

BASE_DIR = "data/Synthetic3/train"
IMG_DIR = os.path.join(BASE_DIR, "JPEGImages")
XML_DIR = os.path.join(BASE_DIR, "Annotations")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(XML_DIR, exist_ok=True)

NUM_IMAGES = 120
IMG_SIZE = 448
CLASSES = ["square", "circle", "triangle"]

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
        
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

for idx in range(NUM_IMAGES):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    num_objects = random.randint(2, 3)
    
    drawn_objects = []
    
    for _ in range(num_objects):
        max_attempts = 100
        success = False
        
        for attempt in range(max_attempts):
            class_id = random.randint(0, 2)
            class_name = CLASSES[class_id]
            
            w = random.randint(40, 80)
            h = w
            cx = random.randint(w, IMG_SIZE - w)
            cy = random.randint(h, IMG_SIZE - h)
            
            x_min = int(cx - w // 2)
            y_min = int(cy - h // 2)
            x_max = int(cx + w // 2)
            y_max = int(cy + h // 2)
            
            new_box = [x_min, y_min, x_max, y_max]
            
            collision = False
            for obj in drawn_objects:
                if calculate_iou(new_box, obj[1]) > 0.0:
                    collision = True
                    break
            
            for obj in drawn_objects:
                dist = np.sqrt((cx - obj[2])**2 + (cy - obj[3])**2)
                if dist < (w/2 + obj[4]/2 + 20):
                    collision = True
                    break
            
            if not collision:
                drawn_objects.append((class_name, new_box, cx, cy, w))
                success = True
                break
                
        if success:
            last_obj = drawn_objects[-1]
            c_name = last_obj[0]
            box = last_obj[1]
            ccx, ccy, ww = last_obj[2], last_obj[3], last_obj[4]
            
            if c_name == "square":
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)
            elif c_name == "circle":
                radius = ww // 2
                cv2.circle(img, (ccx, ccy), radius, (0, 255, 0), -1)
            elif c_name == "triangle":
                pts = np.array([[ccx, box[1]], [box[0], box[3]], [box[2], box[3]]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts], (255, 0, 0))

    img_name = f"synth_{idx:03d}.jpg"
    cv2.imwrite(os.path.join(IMG_DIR, img_name), img)
    
    xml_name = f"synth_{idx:03d}.xml"
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = img_name
    
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(IMG_SIZE)
    ET.SubElement(size, "height").text = str(IMG_SIZE)
    ET.SubElement(size, "depth").text = "3"
    
    for obj in drawn_objects:
        obj_node = ET.SubElement(annotation, "object")
        ET.SubElement(obj_node, "name").text = obj[0]
        
        bndbox = ET.SubElement(obj_node, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj[1][0])
        ET.SubElement(bndbox, "ymin").text = str(obj[1][1])
        ET.SubElement(bndbox, "xmax").text = str(obj[1][2])
        ET.SubElement(bndbox, "ymax").text = str(obj[1][3])
    
    tree = ET.ElementTree(annotation)
    tree.write(os.path.join(XML_DIR, xml_name))

print(f"[SUCCESS] Generated {NUM_IMAGES} non-overlapping multi-object images in: {BASE_DIR}")