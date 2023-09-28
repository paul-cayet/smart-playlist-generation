from typing import Union, List, Dict
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
    ) -> np.ndarray:
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
            if feature_processor
            else IdentityFeatureProcessor()
        )

        if saving_folder is None:
            raise ValueError(
                "Either saving_folder or self.saving_folder must be specified"
            )

        # generate the features
        output_data = []
        for sound_filename in files:
            print(f"Processing file\n\t{sound_filename}")
            features = self.generate_features(sound_filename, **kwargs)
            # transforming the features before storing them
            features = feature_processor.transform(features)

            processed_filename = simple_string_processing(sound_filename)
            output_data.append(
                dict(name=processed_filename, features=features)
            )

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
        print("filename", filename)
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

        return [[1]]


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

        mfccs = librosa.feature.mfcc(
            y=data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length
        )

        return mfccs
