"""
Created on Tue May 13 14:58:24 2025

@author: Matéo Petitet for OFB/Parc Naturel Marin de Martinique
"""
# -*- coding: utf-8 -*-
import os
import json
import cv2
from tqdm import tqdm

# 1) Paramètres
masks_dir  = "/home/mateo/ssd_bis/datasets_coco/Fish4Knowledge-Ground_Truth/images"
output_json = "annotations_coco.json"

# 2) Initialisation du dict COCO
coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "fish", "supercategory": "none"}
    ]
}

ann_id = 1  # compteur global d'annotations
img_id = 1

# 3) Parcours des images
for fname in tqdm(sorted(os.listdir(masks_dir))):
    if not fname.lower().endswith((".jpg", ".png")):
        continue

    # --- a) Lire l'image pour récupérer width/height
    img_path = os.path.join(masks_dir, fname)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Enregistrer l'entrée "images"
    coco["images"].append({
        "id": img_id,
        "file_name": fname,
        "width": w,
        "height": h
    })

    # --- b) Charger le masque binaire correspondant
    mask_path = os.path.join(masks_dir, fname)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, bin_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # --- c) Trouver la bbox du mask
    ys, xs = bin_mask.nonzero()
    if len(xs) == 0:
        # Pas de poisson détecté, on passe
        img_id += 1
        continue

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bbox = [
        int(x_min),
        int(y_min),
        int(x_max - x_min + 1),
        int(y_max - y_min + 1)
    ]

    # --- d) (Optionnel) extraire un polygone de segmentation
    # contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # segmentation = []
    # for cnt in contours:
    #     polygon = cnt.flatten().tolist()
    #     if len(polygon) >= 6:  # au moins 3 points
    #         segmentation.append(polygon)

    # --- e) Enregistrer l’annotation
    coco["annotations"].append({
        "id": ann_id,
        "image_id": img_id,
        "category_id": 1,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
        # "segmentation": segmentation
    })

    ann_id += 1
    img_id += 1

# 5) Sauvegarde du JSON COCO
with open(output_json, "w") as f:
    json.dump(coco, f, indent=2, ensure_ascii=False)

print(f"Fichier COCO généré dans {output_json}")
