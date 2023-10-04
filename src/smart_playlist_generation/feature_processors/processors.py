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
        compression_ratio: int = None,
    ):
        self.compression_ratio = compression_ratio

    def transform(self, data):
        if self.compression_ratio:
            data = self.average_features_along_time(
                data, self.compression_ratio
            )
        else:
            raise NotImplementedError("Not implemented yet")

        return data.tolist()

    @staticmethod
    def average_features_along_time(
        features: np.ndarray,
        compression_ratio: int = None,
    ):
        assert (
            compression_ratio >= 0 and compression_ratio < 1
        ), "The compression ratio must be in [0,1)"
        if compression_ratio == 0:
            return features
        N = int(round(compression_ratio * (features.shape[1] - 1)))

        out = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones((N,)) / N, mode="valid"),
            axis=1,
            arr=features,
        )
        return out
