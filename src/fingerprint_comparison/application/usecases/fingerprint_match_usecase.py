import concurrent.futures
import os
from typing import List

import cv2

from src.fingerprint_comparison.domain.entities.fingerprint import Fingerprint
from src.fingerprint_comparison.domain.strategies.fingerprint_comparison_strategy import (
    FingerprintComparisonStrategyComparisonResult,
)
from src.fingerprint_comparison.infrastructure.strategies.opencv_sift_comparison_strategy import (
    OpenCvSiftComparisonStrategy,
)
from src.shared.domain.usecases.usecase import Usecase


class FingerprintMatchUsecase(Usecase):

    def execute(self, to_find: str, to_match_in: str) -> None:

        strategies = [
            OpenCvSiftComparisonStrategy(),
        ]

        find_image = to_find.split("/")[-1]

        to_find_fingerprint = Fingerprint(
            {
                "image": cv2.imread(to_find),
                "filename": find_image,
            }
        )

        print(f"Image to match: {to_find_fingerprint.filename}")

        print(f"Strategies: {strategies}")

        best_result_for_strategy: List[
            FingerprintComparisonStrategyComparisonResult
        ] = []

        for strategy in strategies:

            print(f"Strategy: {strategy.__getattribute__('__class__').__name__}")

            best_score = 0
            best_image = None
            best_find_image = None
            best_kp1 = None
            best_kp2 = None
            best_mp = None

            fingerprints = os.listdir(to_match_in)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(
                    lambda fingerprint: strategy.compare(
                        to_find_fingerprint,
                        Fingerprint(
                            {
                                "image": cv2.imread(f"{to_match_in}/{fingerprint}"),
                                "filename": fingerprint,
                            }
                        ),
                    ),
                    fingerprints,
                )

                for result in results:
                    score, image, find_image, kp1, kp2, mp = result

                    if score > best_score:
                        best_score = score
                        best_image = image
                        best_find_image = find_image
                        best_kp1 = kp1
                        best_kp2 = kp2
                        best_mp = mp

            best_result_for_strategy.append(
                (best_score, best_image, best_find_image, best_kp1, best_kp2, best_mp),
            )

        for best_result in best_result_for_strategy:
            best_score, best_image, best_find_image, best_kp1, best_kp2, best_mp = (
                best_result
            )

            print(f"Image to find: {to_find_fingerprint.filename}")
            print(f"Best score: {best_score}")
            print(f"Best image: {best_find_image}")

        for idx, (best_result) in enumerate(best_result_for_strategy):
            best_score, best_image, best_find_image, best_kp1, best_kp2, best_mp = (
                best_result
            )

            strategy_name = strategies[idx].__getattribute__("__class__").__name__

            cv2.namedWindow(f"Result {idx + 1} - {strategy_name}", cv2.WINDOW_NORMAL)

            result = cv2.drawMatches(
                to_find_fingerprint.image,
                best_kp1,
                best_image,
                best_kp2,
                best_mp,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            result = cv2.resize(
                result, (800, 800), interpolation=cv2.INTER_CUBIC, fx=0.5, fy=0.5
            )

            cv2.imshow(f"Result {idx} - {strategy_name}", result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
