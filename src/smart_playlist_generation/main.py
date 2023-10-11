from smart_playlist_generation.feature_generation import (
    MelCepstralFeatureGenerator,
)
from smart_playlist_generation.feature_processors import (
    AverageFeatureProcessor,
)
from smart_playlist_generation.optimizers import (
    all_min_transitions,
    optimize_playlist_schedule,
    generate_playlist,
)
from smart_playlist_generation.utils import JsonlFeatureStore
from os import listdir
from os.path import isfile, join
import soundfile as sf
import numpy as np
import argparse


def create_new_features(
    music_folderpath: str,
    saving_folderpath: str,
    saving_filename: str,
    n_mfccs: int,
    compression_ratio: float,
):
    onlyfiles = [
        f
        for f in listdir(music_folderpath)
        if (isfile(join(music_folderpath, f)) and ".ipynb" not in f)
    ]
    music_files = [music_folderpath + x for x in onlyfiles]

    feature_generator = MelCepstralFeatureGenerator(saving_folderpath)
    feature_processor = AverageFeatureProcessor(
        compression_ratio=compression_ratio
    )

    feature_generator.process_batch(
        music_files,
        feature_processor=feature_processor,
        saving_filename=saving_filename,
        n_mfcc=n_mfccs,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--music_folderpath", help="where the music is stored")
    parser.add_argument(
        "--saving_folderpath", help="where the features will be stored"
    )  # noqa: E501
    parser.add_argument(
        "--features_saving_filename",
        help="names of feature file",
        default="features.jsonl",
        required=False,
    )
    parser.add_argument(
        "--n_mfccs",
        help="number of MFCC coefficients",
        default=12,
        required=False,
    )
    parser.add_argument(
        "--generate_new_features",
        help="whether the features were already generated",  # noqa: E501
        default=True,
        required=False,
    )
    parser.add_argument(
        "--compression_ratio",
        help="how much to compress the features (e.g. 0.8 reduces by 80%)",  # noqa: E501
        default=0.85,
        required=False,
    )
    parser.add_argument(
        "--window_size",
        help="how many samples to consider for a transition segment",  # noqa: E501
        default=80,
        required=False,
    )
    parser.add_argument(
        "--max_optimization_time",
        help="maximum time allocated for running the optimization algorithm",  # noqa: E501
        default=10,
        required=False,
    )
    parser.add_argument(
        "--final_playlist_name",
        help="name of the generated playlist",
        default="new_playlist.wav",
        required=False,
    )
    parser.add_argument(
        "--sample_rate",
        help="sample rate for the music files",
        default=22050,
        required=False,
    )

    args = parser.parse_args()

    # creating new features
    if args.generate_new_features:
        create_new_features(args.music_folderpath, args.compression_ratio)

    # finding the mininmum inter-music transitions
    data = JsonlFeatureStore.load_data(
        args.saving_folderpath + args.features_saving_filename
    )
    min_transitions = all_min_transitions(data, args.window_size)

    # optimizing and generating the playlist
    optimized_schedule = optimize_playlist_schedule(
        data, min_transitions, max_processing_time=args.max_optimization_time
    )

    music_list = generate_playlist(
        optimized_schedule[:3], args.music_folderpath
    )
    playlist = np.concatenate(music_list)

    sf.write(
        args.saving_folderpath + args.final_playlist_name,
        playlist,
        args.sample_rate,
        "PCM_24",
    )
