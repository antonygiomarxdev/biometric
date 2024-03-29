type FingerprintArgs = {
    "image": Image,
    "filename": str,
    "score": int,
}


class Fingerprint:
    def __init__(
        self,
        args: FingerprintArgs,
    ) -> None:
        self.image = args["image"]
        self.filename = args["filename"]
        self.score = args["score"]
