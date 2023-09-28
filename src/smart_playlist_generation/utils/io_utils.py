from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import librosa
import json


def read_file(
    path: str,
    sr: Optional[float] = 22050,
    mono: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, float]:
    """Load an audio file as a floating point time series.

    Audio will be automatically resampled to the given rate
    (default ``sr=22050``).

    To preserve the native sampling rate of the file, use ``sr=None``.

    Parameters
    ----------
    path : string, path to the input file.

    sr : number > 0 [scalar]
        target sampling rate

        'None' uses the native sampling rate

    mono : bool
        convert signal to mono

    dtype : numeric type
        data type of ``y``

    Returns
    -------
    y : np.ndarray [shape=(n,) or (..., n)]
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    """
    y, sr = librosa.load(
        path,
        sr=sr,
        mono=mono,
        **kwargs,
    )
    return y, sr


class BaseFeatureStore:
    @staticmethod
    def load_data(path: str):
        raise NotImplementedError("abstract class")

    @staticmethod
    def write_data(data: Any, path: str):
        raise NotImplementedError("abstract class")

    @staticmethod
    def _update_data(data: Any, new_data: Any):
        raise NotImplementedError("abstract class")

    def append_data(self, new_data: Any, path: str):
        data = self.load_data(path)
        data = self._update_data(data, new_data)
        self.write_data(data, path)


class JsonlFeatureStore:
    SEP = "\n"

    @staticmethod
    def load_data(path: str):
        with open(path, "r") as f:
            data = f.read().split(JsonlFeatureStore.SEP)
            data = [json.loads(row) for row in data]
        return data

    @staticmethod
    def write_data(data: List[Dict], path: str):
        with open(path, "w") as f:
            data = [json.dumps(row) for row in data]
            str_data = JsonlFeatureStore.SEP.join(data)
            f.write(str_data)

    @staticmethod
    def append_data(new_data: List[Dict], path: str):
        data = JsonlFeatureStore.load_data(path)
        data = data + new_data
        JsonlFeatureStore.write_data(data, path)
