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

    music_folderpath = "/home/paul/Documents/personal/python_projects/smart-playlist-generation/data/music_files/phonk/"  # noqa: E501
    saving_folderpath = "/home/paul/Documents/personal/python_projects/smart-playlist-generation/data/"  # noqa: E501
    features_saving_filename = "features_phonk.jsonl"
    n_mfccs = 12
    generate_new_features = False
    compression_ratio = 0.85
    window_size = 80
    max_optimization_time = 10
    final_playlist_name = "new_playlist.wav"
    sample_rate = 22050

    # creating new features
    if generate_new_features:
        create_new_features(music_folderpath, compression_ratio)

    # finding the mininmum inter-music transitions
    data = JsonlFeatureStore.load_data(
        saving_folderpath + features_saving_filename
    )
    min_transitions = all_min_transitions(data, window_size)

    # optimizing and generating the playlist
    optimized_schedule = optimize_playlist_schedule(
        data, min_transitions, max_processing_time=max_optimization_time
    )

    music_list = generate_playlist(optimized_schedule[:3], music_folderpath)
    playlist = np.concatenate(music_list)

    sf.write(
        saving_folderpath + final_playlist_name,
        playlist,
        sample_rate,
        "PCM_24",
    )
