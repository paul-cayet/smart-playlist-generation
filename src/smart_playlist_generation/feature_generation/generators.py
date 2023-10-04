from typing import Union, List, Dict, Tuple
import numpy as np
import librosa
from smart_playlist_generation.utils import JsonlFeatureStore
from smart_playlist_generation.feature_processors import (
    BaseFeatureProcessor,
    IdentityFeatureProcessor,
)
from smart_playlist_generation.utils import read_file


DEFAULT_FS_FILENAME = "test.jsonl"
BASE_SAVING_METHOD = "jsonl"


def simple_string_processing(s: str):
    s = s.split("/")[-1].split(".")[0]
    print(s)
    s = "".join([x.lower() for x in s if (x.isalnum() or x == " ")])
    s = s.replace("  ", " ").replace(" ", "_")
    return s


def quantile_normalize(x, q=0.9):
    qmin = np.quantile(x, q=(1 - q))
    qmax = np.quantile(x, q=q)
    return (x - qmin) / (qmax - qmin)


def custom_normalization(x, q=0.9):
    """we normalize the vector x by doing 3 normalization
    in 3 different groups: x<-100, -100<x<100 and 100<x
    """
    min_elements = 20
    lb, ub = -100, 100
    low_idx, up_idx, mid_idx = (x < lb), (x > ub), ((x > lb) & (x < ub))
    for range_ in (low_idx, mid_idx, up_idx):
        if len(x[range_] > min_elements):
            x[range_] = quantile_normalize(x[range_], q=q)

    return x


def reshape_features(
    features: np.ndarray, window_size: int, normalize: bool = False, verbose=1
):
    """We reshape the array from (n_features, window_size*n_windows)
    to (n_windows,n_features*window_size)
    """
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    n_windows = features.shape[1] // window_size
    if features.shape[1] % window_size != 0:
        new_size = n_windows * window_size
        if verbose > 0:
            print(
                f"{features.shape[1]-new_size} elements will be lost due to unequal array split"  # noqa: E501
            )
        features = features[:, :new_size]
        # print("Number of input features must be a multiple of window_size")

    split_features = np.split(features, n_windows, axis=1)
    features = np.array(split_features).transpose((0, 2, 1))
    features = features.reshape((n_windows, -1))

    if normalize:
        features = np.apply_along_axis(
            custom_normalization, arr=features, axis=1
        )
    return features


def save_generated_features(
    features: List[Dict], path: str, method: str = None
):
    print(path)
    method = method if method else BASE_SAVING_METHOD
    if method == "jsonl":
        # print('file stored')
        JsonlFeatureStore.write_data(features, path)
    else:
        raise NotImplementedError(
            f"Saving method {method} is not yet implemented"
        )


class BaseFeatureGenerator:
    def __init__(self, saving_folder: str = None):
        """Initializes the Feature Generator
        Parameters
        ----------
        saving_folder: str
            Location to store generated feature to.
        """
        self.saving_folder = saving_folder

    def generate_features(
        self,
        data: Union[str, np.ndarray],
        sr: int = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate the features representing the given data.
        Parameters
        ----------
        data : Union[str; np.ndarray]
            if str, will attempt to read the file at the specified location
            if array, will directly generate the features
        sr : number > 0 [scalar]
            file sampling rate. If None, data must be of type int

        Returns
        -------
        features: np.ndarray
            The generated features
        data_len: int
            Length of the original sound file
        """
        raise NotImplementedError("abstract class")

    # def process_single(
    #             self,
    #             data: Union[str, np.ndarray],
    #             sr: int=None,
    #             feature_processor: BaseFeatureProcessor=None,
    #             saving_folder: str=None,
    #             saving_name: str=None,
    #             saving_method: str=None,
    #             **kwargs,
    #     ):
    #     path = saving_folder if saving_folder else self.saving_folder
    #     feature_processor = feature_processor if feature_processor else IdentityFeatureProcessor() # noqa: E501
    #     if path is None:
    #         raise ValueError(
    #               'Either saving_folder or path must be specified'
    #           )
    #     if isinstance(data,str) == False and saving_name is None:
    #         raise ValueError(
    #            'If the data is an array, the saving_name must be specified'
    #          )

    #     # generate the features
    #     features = self.generate_features(data,sr,**kwargs)

    #     output_data = dict(
    #                     name=saving_name,
    #                     features=features
    #                 )

    #     # save the features
    #     self.save_features(
    #             features,
    #             saving_folder=saving_folder,
    #             filename=saving_name,
    #             method=saving_method,
    #     )

    def process_batch(
        self,
        files: List[str],
        feature_processor: BaseFeatureProcessor = None,
        saving_folder: str = None,
        saving_filename: str = None,
        saving_method: str = None,
        **kwargs,
    ):
        """Generate features for multiple files and save them
        files : List[str]
            List of file locations to use to generate features
        """
        saving_folder = saving_folder if saving_folder else self.saving_folder
        feature_processor = (
            feature_processor
            if feature_processor is not None
            else IdentityFeatureProcessor()
        )

        if saving_folder is None:
            raise ValueError(
                "Either saving_folder or self.saving_folder must be specified"
            )

        # generate the features
        output_data = []
        for sound_filename in files:
            try:
                print(f"Processing file\n\t{sound_filename}")
                features, data_len = self.generate_features(
                    sound_filename, **kwargs
                )
                # transforming the features before storing them

                features = feature_processor.transform(features)
                print(
                    f"""Feature length after processing={(len(features), len(features[0]))}, ratio={data_len//len(features[0])}\n"""  # noqa: E501
                )

                # sound_filename = simple_string_processing(sound_filename)
                output_data.append(
                    dict(name=sound_filename, features=features)
                )
                # print(f"features shape = {(len(features),len(features[0]))}")
            except Exception as e:
                print(f"Failed processing, error was {e}")
                raise e
        # save the features
        self.save_features(
            output_data,
            saving_folder=saving_folder,
            filename=saving_filename,
            method=saving_method,
        )

    def save_features(
        self,
        features: List[Dict],
        saving_folder: str = None,
        filename: str = None,
        method: str = None,
    ):
        """
        Either saving folder or self.saving_folder must be defined
        If filename is not defined, a default name will be used.
        """
        saving_path = saving_folder if saving_folder else self.saving_folder
        filename = filename if filename else DEFAULT_FS_FILENAME
        if saving_path is None:
            raise ValueError(
                "Either saving folder or self.saving_folder must be defined"
            )
        saving_path = saving_path + filename

        save_generated_features(
            features=features, path=saving_path, method=method
        )


class ConstantFeatureGenerator(BaseFeatureGenerator):
    def generate_features(
        self,
        data: Union[str, np.ndarray],
        sr: int = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate the features representing the given data.
        Parameters
        ----------
        data : Union[str; np.ndarray]
            if str, will attempt to read the file at the specified location
            if array, will directly generate the features
        sr : number > 0 [scalar]
            file sampling rate. If None, data must be of type str

        Returns
        -------
        features: np.ndarray
            The generated features
        """
        if isinstance(data, str):
            path = data
            data, sr = read_file(path=path, sr=sr, **kwargs)
        elif sr is None:
            raise ValueError(
                "If data is an array, the sampling rate sr is also required"
            )

        return [[1]], len(data)


class MelCepstralFeatureGenerator(BaseFeatureGenerator):
    def generate_features(
        self,
        data: Union[str, np.ndarray],
        sr: int = None,
        n_mfcc: int = 20,
        hop_length: int = 512,
        **kwargs,
    ) -> np.ndarray:
        """Generate the features representing the given data.
        Parameters
        ----------
        data : Union[str; np.ndarray]
            if str, will attempt to read the file at the specified location
            if array, will directly generate the features
        sr : number > 0 [scalar]
            file sampling rate. If None, data must be of type str

        Returns
        -------
        features: np.ndarray
            The generated features
        """
        if isinstance(data, str):
            path = data
            data, sr = read_file(path=path, sr=sr, **kwargs)
        elif sr is None:
            raise ValueError(
                "If data is an array, the sampling rate sr is also required"
            )
        print(f"File length={len(data)}, sr={sr}")
        mfccs = librosa.feature.mfcc(
            y=data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length
        )
        print(
            f"""Feature length before processing={mfccs.shape},
            ratio={len(data)//mfccs.shape[1]}"""
        )
        return mfccs, len(data)
