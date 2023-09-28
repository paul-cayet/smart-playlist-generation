import numpy as np


class BaseFeatureProcessor:
    def __init__(self):
        raise NotImplementedError("abstract class")

    def transform(self, data):
        raise NotImplementedError("abstract class")


class IdentityFeatureProcessor(BaseFeatureProcessor):
    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return data


class AverageFeatureProcessor(BaseFeatureProcessor):
    def __init__(
        self,
        new_time_length: int = None,
        new_feature_size: int = None,
    ):
        self.new_time_length = new_time_length
        self.new_feature_size = new_feature_size

    def transform(self, data):
        if self.new_time_length:
            data = self.average_features_along_time(data, self.new_time_length)
        if self.new_feature_size:
            raise NotImplementedError("Not implemented yet")

        return data.tolist()

    @staticmethod
    def average_features_along_time(features: np.ndarray, new_length: int):
        N = features.shape[1] + 1 - new_length
        if N < 1:
            raise ValueError(
                "The new length cannot be greater than the previous length"
            )

        out = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones((N,)) / N, mode="valid"),
            axis=1,
            arr=features,
        )
        return out


temp = np.random.random((6, 8000))
AverageFeatureProcessor.average_features_along_time(temp, new_length=3)
