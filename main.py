import os

import cv2

best_score = 0
image = None
filename = None
kp1, kp2, mp = None, None, None

sample = cv2.imread("SOCOFing/Altered/Altered-Hard/150__M_Right_index_finger_Obl.BMP")

type BestScore = int
type Image = cv2.Mat
type Filename = str
type KeyPoints = list[cv2.KeyPoint]
type MatchPoints = cv2.DMatch

type GetCoincidence = (BestScore, Image, Filename, KeyPoints, KeyPoints, MatchPoints)


def get_coincidence() -> GetCoincidence:
    counter = 0

    global best_score, image, filename, kp1, kp2, mp
    for file in [file for file in os.listdir("SOCOFing/Real")]:
        if counter % 10 == 0:
            print(counter)
            print(file)
        counter += 1

        fingerprint_image = cv2.imread("SOCOFing/Real/" + file)
        sift = cv2.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        matches = cv2.FlannBasedMatcher(
            {
                "algorithm": 1,
                "trees": 10,
            },
            {},
        ).knnMatch(descriptors_1, descriptors_2, k=2)

        match_points = []

        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                match_points.append(p)

        keypoints = 0
        if len(keypoints_1) < len(keypoints_2):
            keypoints = len(keypoints_1)
        else:
            keypoints = len(keypoints_2)

        if len(match_points) / keypoints * 100 > best_score:
            best_score = len(match_points) / keypoints * 100
            image = fingerprint_image
            filename = file
            kp1, kp2, mp = keypoints_1, keypoints_2, match_points

    return best_score, image, filename, kp1, kp2, mp


if __name__ == "__main__":
    best_score, image, filename, kp1, kp2, mp = get_coincidence()

    print("Best match: " + filename)
    print("Score: " + str(best_score))

    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result = cv2.resize(result, None, fx=4, fy=4)
    cv2.imshow("Resultado", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
