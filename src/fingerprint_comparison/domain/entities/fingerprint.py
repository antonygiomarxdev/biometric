from src.fingerprint_comparison.domain.types import Image

type FingerprintArgs = {
    "image": Image,
    "filename": str,
}


class Fingerprint:
    def __init__(
        self,
        args: FingerprintArgs,
    ) -> None:
        self.image = args["image"]
        self.filename = args["filename"]
