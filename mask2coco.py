"""
Created on Tue May 13 14:58:24 2025

@author: Matéo Petitet for OFB/Parc Naturel Marin de Martinique
"""
# -*- coding: utf-8 -*-
import os
import json
import cv2
from tqdm import tqdm

# --- Paramètres
masks_dir   = "/home/mateo/ssd_bis/datasets_coco_ok/DeepFish/valid/"
output_json = "annotations_coco_multi.json"

# --- Initialisation COCO
coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "fish", "supercategory": "none"}
    ]
}

ann_id = 1
img_id = 1

# --- Parcours des images
for fname in tqdm(sorted(os.listdir(masks_dir))):
    if not fname.lower().endswith((".jpg", ".png")):
        continue

    # 1) Lire image pour largeur/hauteur
    img = cv2.imread(os.path.join(masks_dir, fname))
    h, w = img.shape[:2]
    coco["images"].append({
        "id": img_id, "file_name": fname, "width": w, "height": h
    })

    # 2) Charger et binariser le masque
    mask_path = os.path.join(masks_dir, fname)
    mask = cv2.imread(os.path.join(masks_dir, fname), cv2.IMREAD_GRAYSCALE)
    _, bin_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # 3) Trouver chaque poisson via les contours
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Ignore les petites zones parasites
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        # 4) Calculer la bbox pour ce contour
        x, y, w_box, h_box = cv2.boundingRect(cnt)

        # # 5) (Optionnel) polygon de segmentation
        # segmentation = cnt.flatten().tolist()
        # if len(segmentation) < 6:
        #     # il faut au moins 3 points (6 valeurs) pour un polygone valide
        #     segmentation = []

        # 6) Ajouter l’annotation COCO
        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [x, y, w_box, h_box],
            "area": w_box * h_box,
            "iscrowd": 0
            #**({"segmentation": [segmentation]} if segmentation else {})
        })
        ann_id += 1

    img_id += 1

# --- Sauvegarde
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2, ensure_ascii=False)

print(f"Annotations COCO multi-objets générées dans {output_json}")

