from dataclasses import dataclass


def compare_features_by_distance(feature1, feature2):
    if abs(0.5 - feature1.value) < abs(0.5 - feature2.value):
        return -1
    elif abs(0.5 - feature1.value) > abs(0.5 - feature2.value):
        return 1
    else:
        return 0


def compare_features(feature1, feature2):
    if feature1.value < feature2.value:
        return -1
    elif feature1.value > feature2.value:
        return 1
    else:
        return 0


@dataclass(frozen=True)
class Feature:
    id: int
    value: float

    def to_dict(self):
        return {
            'id': self.id,
            'value': self.value
        }
