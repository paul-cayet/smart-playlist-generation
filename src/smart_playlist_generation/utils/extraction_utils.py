from smart_playlist_generation.utils import read_file
import numpy as np
from scipy.ndimage import shift
import librosa


def extract_section(
    data,
    section_idx: int,
    n_sections: int,
    n_sections_to_select=1,
    before=True,
):
    """before means we want to select the music that is
    before the section idx
    """
    n = len(data)
    section_len = int(round(n / n_sections))
    if before:
        start_idx = max(0, (section_idx - n_sections_to_select) * section_len)
        end_idx = min(n, (section_idx) * section_len)
    else:  # is beginning
        start_idx = max(0, (section_idx) * section_len)
        end_idx = min(n, (section_idx + n_sections_to_select) * section_len)
    print("extracting idx from", start_idx, end_idx)
    return data[start_idx:end_idx]


def generate_fade_curve(
    length: int,
    null_length: int,
    fade_type="linear",
    is_end: bool = True,
    sigmoid_curvature: float = 0.01,
):
    if fade_type == "linear":
        fade_curve = np.linspace(1.0, 0.0, length)
    elif fade_type == "sigmoid":
        alpha = 2 * np.log(1 / (1 - sigmoid_curvature) - 1) / length
        fade_curve = np.linspace(0, length, length)
        fade_curve = 1 / (1 + np.exp(-alpha * (fade_curve - length / 2)))

    if is_end:
        fade_curve = np.concatenate((fade_curve, np.zeros((null_length))))
    if not is_end:
        fade_curve = np.concatenate((np.zeros((null_length)), 1 - fade_curve))

    return fade_curve


def apply_fade(
    audio,
    sr,
    duration=4.0,
    null_duration=1.0,
    is_end: bool = True,
    fade_type="sigmoid",
    sigmoid_curvature: float = 0.01,
):
    # convert to audio indices (samples)
    length = int(duration * sr)
    null_length = int(null_duration * sr)
    audio_len = audio.shape[0]
    end = audio_len if is_end else length + null_length
    start = audio_len - length - null_length if is_end else 0

    # compute fade out curve
    fade_curve = generate_fade_curve(
        length, null_length, fade_type, is_end, sigmoid_curvature
    )

    audio = audio.copy()
    print(
        f"start: {start}, end: {end} ({end-start}), curve: {len(fade_curve)}"
    )
    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve
    return audio


def apply_crossfade(
    audio1,
    audio2,
    sr,
    duration=4.0,
    null_duration=1.0,
    fade_type="linear",
    sigmoid_curvature: float = 0.01,
):
    """audio1 is crossfaded into audio2 with a linear fade"""
    # apply fades
    audio1 = apply_fade(
        audio1, sr, duration, null_duration, True, fade_type, sigmoid_curvature
    )
    audio2 = apply_fade(
        audio2,
        sr,
        duration,
        null_duration,
        False,
        fade_type,
        sigmoid_curvature,
    )

    # stitch the files together
    overlap = int((duration + null_duration) * sr)
    audio1_len, audio2_len = audio1.shape[0], audio2.shape[0]
    new_audio = np.zeros(
        (audio1_len + audio2_len - overlap), dtype=audio1.dtype
    )
    new_audio[:audio1_len] = audio1
    new_audio[audio1_len - overlap :] += audio2

    return new_audio


def sync_on_beat(audio1, audio2, sr):
    hop_length = 512
    tempo1, beats1 = librosa.beat.beat_track(y=audio1, sr=sr)
    tempo2, beats2 = librosa.beat.beat_track(y=audio2, sr=sr)
    print(f"Audio 1 tempo={tempo1:.0f}, audio 2 tempo={tempo2:.0f}")

    # compute dynamic warping
    D, wp = librosa.sequence.dtw(beats1, beats2, subseq=True)
    shift_coeff = np.argmin(D[-1, :])
    audio2 = shift(audio2, shift_coeff * hop_length, cval=0)
    return audio1, audio2
