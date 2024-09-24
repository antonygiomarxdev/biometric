class Fingerprint:
    def __init__(self, id: str, minutiae_points: list):
        self.id = id
        self.minutiae_points = minutiae_points

    def __repr__(self):
        return f"Fingerprint(id={self.id})"
        