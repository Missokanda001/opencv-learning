import cv2
import numpy as np
import os

# Your reference images (1 per class)
REFERENCES = {
    "bag":    r"D:\project_envs\endoscopy-pano\opencv-learning\module02-ch09-object-recognition\bag.jpg",
    "dress":  r"D:\project_envs\endoscopy-pano\opencv-learning\module02-ch09-object-recognition\dress.jpg",
    "shoes":  r"D:\project_envs\endoscopy-pano\opencv-learning\module02-ch09-object-recognition\shoes.jpg",
}

def compute_sift(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des, img

def match_objects(test_image_path):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    
    test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    kp_test, des_test = sift.detectAndCompute(test_img, None)
    
    best_class = None
    best_score = 0
    
    for class_name, ref_path in REFERENCES.items():
        kp_ref, des_ref, _ = compute_sift(ref_path)
        matches = bf.knnMatch(des_test, des_ref, k=2)
        
        # Lowe's ratio test
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        score = len(good)
        
        print(f"{class_name}: {score} good matches")
        if score > best_score:
            best_score = score
            best_class = class_name
    
    return best_class, best_score

if __name__ == "__main__":
    test_img = r"D:\project_envs\endoscopy-pano\opencv-learning\module02-ch09-object-recognition\test.jpg"
    result, score = match_objects(test_img)
    print(f"\nPredicted object: {result} (score: {score})")