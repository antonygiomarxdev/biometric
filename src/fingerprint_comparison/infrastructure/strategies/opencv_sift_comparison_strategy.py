from typing import Sequence

import cv2
from numpy import ndarray

from src.fingerprint_comparison.domain.entities.fingerprint import Fingerprint
from src.fingerprint_comparison.domain.strategies.fingerprint_comparison_strategy import (
    FingerprintComparisonStrategy,
    FingerprintComparisonStrategyComparisonResult,
)

type Score = float
type Image = ndarray
type Filename = str
type KeyPoints = Sequence[cv2.KeyPoint]
type MatchPoints = Sequence[cv2.DMatch]


class OpenCvSiftComparisonStrategy(FingerprintComparisonStrategy):
    def compare(
        self, fingerprint1: Fingerprint, fingerprint2: Fingerprint
    ) -> FingerprintComparisonStrategyComparisonResult:

        score: Score = 0
        image: Image = ndarray(0, dtype=int)
        filename: Filename = ""
        kp1: KeyPoints = []
        kp2: KeyPoints = []
        mp: MatchPoints = []

        sift = cv2.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(fingerprint1.image, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint2.image, None)

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

        if len(match_points) / keypoints * 100 > score:
            score = len(match_points) / keypoints * 100
            image = fingerprint2.image
            filename = fingerprint2.filename
            kp1, kp2, mp = keypoints_1, keypoints_2, match_points

        return score, image, filename, kp1, kp2, mp
