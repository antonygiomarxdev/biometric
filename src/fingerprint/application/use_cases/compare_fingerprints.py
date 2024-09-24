from ..services.fingerprint_service import FingerprintService

class CompareFingerprints:
    def __init__(self, service: FingerprintService):
        self.service = service

    def execute(self, fingerprint_1, fingerprint_2):
        return self.service.compare_fingerprints(fingerprint_1, fingerprint_2)
        