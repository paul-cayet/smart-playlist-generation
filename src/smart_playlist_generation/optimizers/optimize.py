from smart_playlist_generation.feature_generation import reshape_features
from smart_playlist_generation.utils import (
    read_file,
    extract_section,
    apply_crossfade,
)
from typing import Dict
import numpy as np
import python_tsp.heuristics as heuristics
import time


def all_min_transitions(
    data,
    window_size: int,
    range1=(0.7, 0.9),
    range2=(0.1, 0.3),
    normalize: bool = True,
    verbose: int = 1,
):
    n_elts = len(data)
    transitions = {}
    for i1, elt1 in enumerate(data):
        if verbose > 0:
            print(f"Done {100*i1/n_elts:.1f}%")
        music1 = reshape_features(
            np.array(elt1["features"]),
            window_size=window_size,
            verbose=verbose - 1,
            normalize=normalize,
        )
        for i2, elt2 in enumerate(data):
            if i1 == i2:
                continue
            music2 = reshape_features(
                np.array(elt2["features"]),
                window_size=window_size,
                verbose=verbose - 1,
                normalize=normalize,
            )
            min_transition = single_min_transition(
                music1, music2, range1, range2
            )
            transitions[(i1, i2)] = min_transition

    return transitions


def single_min_transition(list1, list2, range1=(0.7, 0.9), range2=(0.1, 0.3)):
    """Find argmin(|list1[i1]-list2[i2]|) iterating
    for i1 from alpha1*len(list1)) and i2 up to alpha2*len(list2)
    list1 and list2 are lists of representation vectors for
    musics 1 and 2
    """
    len1, len2 = len(list1), len(list2)
    start1, end1 = int(round(range1[0] * len1)), int(round(range1[1] * len1))
    start2, end2 = int(round(range2[0] * len2)), int(round(range2[1] * len2))

    min_elt = (float("inf"), None)
    for i1 in range(start1, end1):
        x1 = list1[i1]
        for i2 in range(start2, end2):
            x2 = list2[i2]
            dist = np.linalg.norm(x1 - x2)
            if dist < min_elt[0]:
                min_elt = (dist, ((i1, len1), (i2, len2)))
    return min_elt


def create_cost_matrix(
    all_transitions: Dict, n_elts: int, default_value: float = -0.0
):
    assert (
        len(all_transitions) == n_elts**2 - n_elts
    ), "Warning, missing values detected"
    res = [[default_value for _ in range(n_elts)] for _ in range(n_elts)]
    for i in range(n_elts):
        for j in range(n_elts):
            if i == j:
                continue
            if (i, j) not in all_transitions:
                print(f"Warning: {(i,j)} not in all_transitions")
                continue
            res[i][j] = all_transitions[(i, j)][0]
    return np.array(res)


def optimize_playlist_schedule(data, transitions, max_processing_time=None):

    dist_matrix = create_cost_matrix(transitions, len(data))
    print("Optimization started...")
    ptime = time.time()
    permutation, distance = heuristics.solve_tsp_simulated_annealing(
        dist_matrix, alpha=0.999, max_processing_time=max_processing_time
    )
    print(
        f"Optimization finished in {time.time()-ptime:.1f}s, min={distance:.1f}"  # noqa: E501
    )

    schedule = []
    for i in range(1, len(permutation)):
        idx1, idx2 = permutation[i - 1], permutation[i]
        _, (info1, info2) = transitions[(idx1, idx2)]
        schedule.append(
            [(data[idx1]["name"], info1), (data[idx2]["name"], info2)]
        )
    return schedule


def generate_playlist(playlist_schedule, music_folder_path: str):
    # add fake final music to the schedule
    final_musicname, (_, final_n_sections) = playlist_schedule[-1][1]
    playlist_schedule.append(
        [(final_musicname, (final_n_sections + 1, final_n_sections))]
    )
    music_list = []

    # 1. add the first music to the transition
    filename, (section_idx, n_sections) = playlist_schedule[0][0]
    music, sr = read_file(music_folder_path + filename)

    section = extract_section(
        music,
        section_idx - 1,
        n_sections,
        n_sections_to_select=section_idx + 1,
        before=True,
    )
    music_list.append(section)

    # 2. add transition + next music to the next transition
    for i in range(len(playlist_schedule) - 1):
        print(f"Processing transition {i+1}/{len(playlist_schedule)-1}")
        _, (section_idx, n_sections) = playlist_schedule[i][0]
        next_filename, (next_section_idx, next_n_sections) = playlist_schedule[
            i
        ][1]
        _, (next_transition_section_idx, _) = playlist_schedule[i + 1][0]
        next_music, sr = read_file(music_folder_path + next_filename)

        # extract the transition sections
        delta = 1
        section1 = extract_section(
            music,
            section_idx - 1,
            n_sections,
            n_sections_to_select=1 + delta,
            before=False,
        )
        section2 = extract_section(
            next_music,
            next_section_idx,
            next_n_sections,
            n_sections_to_select=next_transition_section_idx
            - next_section_idx
            - delta,
            before=False,
        )
        # generate the transition
        # section1,section2 = sync_on_beat(section1,section2, sr)
        # // TODO: Understand why it messes the code
        transition_section = apply_crossfade(
            section1,
            section2,
            sr,
            duration=7.0,
            null_duration=0,
            fade_type="sigmoid",
            sigmoid_curvature=0.05,
        )
        music_list.append(transition_section)

        music = next_music
        section_idx = next_section_idx
        n_sections = next_n_sections

    return music_list
