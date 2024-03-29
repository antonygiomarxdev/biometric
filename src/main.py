from src.fingerprint_comparison.application.usecases.fingerprint_match_usecase import (
    FingerprintMatchUsecase,
)

if __name__ == "__main__":

    sample = "../SOCOFing/Altered/Altered-Hard/4__M_Left_ring_finger_CR.BMP"

    fingerprint_match_usecase = FingerprintMatchUsecase()

    fingerprint_match_usecase.execute(sample, "../SOCOFing/Real")
