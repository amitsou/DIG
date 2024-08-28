"""
Execution path: project/src/dynamic_image.py
usage example: python3 dynamic_image.py -i "/media/alex/EVOSSD/PhD/Code_base/egocentric_vision/Datasets/Frames/BON-dataset/" -m "get_dyn_img"
usage example: python3 dynamic_image.py -i "/media/alex/EVOSSD/PhD/Code_base/egocentric_vision/Datasets/BON-dataset/" -m 'get_frames'
usage example: python3 dynamic_image.py -i "/media/alex/EVOSSD/PhD/Code_base/egocentric_vision/Datasets/CharadesEgo/" -m 'get_dyn_img'
"""

import argparse
import os
import time

from utils.utils import (
    blockPrint,
    calculate_execution_time,
    create_multiple_dirs,
    get_videos,
)
from utils.visual_utils import create_dynamic_images, extract_video_frames


def parse_args():
    """Parse the command line arguments
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Input path directory containing the videos"
    )
    parser.add_argument(
        "-p", "--print", help="Flag to print messages", action="store_true"
    )
    parser.add_argument("-m", "--mode", help="Mode: [get_frames, get_dyn_img]")
    return parser.parse_args()


def process_datasets(input_dir: str, mode: str):
    extensions = {
        "get_frames": [".mp4", ".MP4"],
        "get_dyn_img": [".jpg", ".JPG", ".png", ".PNG"],
    }

    if "BON" in input_dir:
        subdirs = ["Barcelona", "Nairobi", "Oxford"]
    elif "Charades" in input_dir:
        """
        Due to the reason that RGB frames are provided in the CharadesEgo dataset there is no need for extracting them.
        Thus, the only thing that needs to be done is to create the dynamic images.
        """
        subdirs = ["CharadesEgo_v1_rgb"]
    else:
        raise ValueError(f"Invalid input directory: {input_dir}")

    for subdir in subdirs:
        tmp_dir = "".join((input_dir, subdir))
        videos = get_videos(tmp_dir, extensions.get(mode, []))

        if mode == "get_dyn_img" and "BON" in tmp_dir:
            output_dir = list(
                map(
                    lambda video: os.path.dirname(
                        video.replace(
                            "Datasets/Frames", "Datasets/Dynamic_Images"
                        ).replace("/Frames", "")
                    ),
                    videos,
                )
            )
        elif mode == "get_dyn_img" and "Charades" in tmp_dir:
            output_dir = list(
                map(
                    lambda video: os.path.dirname(
                        video.replace("Datasets", "Datasets/Dynamic_Images")
                    ),
                    videos,
                )
            )
        elif mode == "get_frames":
            output_dir = list(
                map(
                    lambda video: os.path.dirname(
                        video.replace("Datasets", "Datasets/Frames")
                    ),
                    videos,
                )
            )

        [create_multiple_dirs(directory) for directory in sorted(list(set(output_dir)))]

        functions = {
            "get_frames": extract_video_frames,
            "get_dyn_img": create_dynamic_images,
        }
        functions[mode](videos, output_dir)


def main():
    start_time = time.time()
    args = parse_args()

    if args.print:
        blockPrint()

    process_datasets(str(args.input), str(args.mode))
    end_time = time.time()
    calculate_execution_time(start_time, end_time)


if __name__ == "__main__":
    main()
