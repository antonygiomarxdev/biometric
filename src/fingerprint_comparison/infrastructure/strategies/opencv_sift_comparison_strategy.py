from src.fingerprint_comparison.domain.strategies.fingerprint_comparison_strategy import (
    FingerprintComparisonStrategy,
    FingerprintComparisonStrategyComparisonResult,
)


class OpenCvSiftComparisonStrategy(FingerprintComparisonStrategy):
    def compare(
        self, fingerprint1, fingerprint2
    ) -> FingerprintComparisonStrategyComparisonResult:
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(fingerprint1.image, None)
        keypoints2, descriptors2 = sift.detectAndCompute(fingerprint2.image, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        return len(good_matches) / min(len(keypoints1), len(keypoints2))
