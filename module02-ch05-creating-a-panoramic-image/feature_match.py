import cv2
import numpy as np

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    # Create a new output image that concatenates the two images together
    output_img = np.zeros((max([rows1,rows2]), cols1+cols2, 3), dtype='uint8')
    output_img[:rows1, :cols1, :] = np.dstack([img1, img1, img1])
    output_img[:rows2, cols1:cols1+cols2, :] = np.dstack([img2, img2, img2])

    # Draw connecting lines between matching keypoints
    for match in matches:
        # Get the matching keypoints for each of the images
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        # Draw a small circle at both co-ordinates and then draw a line
        radius = 4
        colour = (0,255,0)   # green
        thickness = 1
        cv2.circle(output_img, (int(x1),int(y1)), radius, colour, thickness)
        cv2.circle(output_img, (int(x2)+cols1,int(y2)), radius, colour, thickness)
        cv2.line(output_img, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), colour, thickness)
    return output_img

if __name__=='__main__':
    # Use your relative path (update if needed)
    img1 = cv2.imread("D:\\project_envs\\endoscopy-pano\\opencv-learning\\module02-ch05-creating-a-panoramic-image\\img1.jpg", 0)
    img2 = cv2.imread("D:\\project_envs\\endoscopy-pano\\opencv-learning\\module02-ch05-creating-a-panoramic-image\\img2.jpg", 0)

    # Safety check for missing images
    if img1 is None or img2 is None:
        print("ERROR: Cannot load image files. Check file path and name!")
        exit()

    # Initialize ORB detector
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Brute Force Matcher (for binary ORB descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # Get top 2 candidate matches for every descriptor (required for ratio test)
    raw_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's Ratio Test: filter out ambiguous / false matches
    good_matches = []
    ratio_threshold = 0.75
    for m, n in raw_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # Draw only filtered reliable matches
    img_result = draw_matches(img1, keypoints1, img2, keypoints2, good_matches[:30])

    cv2.imshow('Filtered ORB Matches (Good matches only)', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()