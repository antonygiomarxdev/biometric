from abc import ABC, abstractmethod

class FingerprintRepository(ABC):
    @abstractmethod
    def compare(self, fingerprint_1, fingerprint_2):
        pass
        