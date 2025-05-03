import os
import cv2
import matplotlib.pyplot as plt

sample = cv2.imread("Altered/Altered-Hard/150__M_Right_index_finger_Obl.BMP")

# Move keypoint detection and descriptor computation for the sample image outside the loop
sift = cv2.SIFT_create()
keypoints_1, des1 = sift.detectAndCompute(sample, None)

best_score = counter = 0
filename = image = kp1 = kp2 = mp = None

for file in os.listdir("Real")[:1000]:
    if counter % 10 == 0:
        print(counter)
        print(file)
    counter += 1

    fingerprint_img = cv2.imread("Real/" + file)
    keypoints_2, des2 = sift.detectAndCompute(fingerprint_img, None)

    # fast library for approx best match KNN
    matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(des1, des2, k=2)

    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = min(len(keypoints_1), len(keypoints_2))
    current_score = len(match_points) / keypoints * 100

    # Print the score for the current file
    print(f"Score for {file}: {current_score}")

    if current_score > best_score:
        best_score = current_score
        filename = file
        image = fingerprint_img
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

    if len(match_points) > 0:
        result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
        result = cv2.resize(result, None, fx=5, fy=5)
        # Uncomment the next line if you want to display using OpenCV
        # cv2.imshow("Result", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Display using Matplotlib
        image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()

print("*****************************************************")
print("Best match:  " + filename)
print("Best score:  " + str(best_score))
