import datetime
import glob
import os

import cv2
import numpy as np
from utils.utils import create_multiple_dirs


def create_dynamic_images(input_dir: str, output_dir: str):
    """Create dynamic images from a list of videos.

    Args:
        input_dir (str): The directory containing the videos.
        output_dir (str): The directory to save the dynamic images.
    """
    video_directories = [os.path.dirname(video) for video in input_dir]
    video_directories = list(set(video_directories))
    video_directories = sorted(video_directories)

    output_dir = list(set(output_dir))
    output_dir = sorted(output_dir)

    frames = []
    for video_dir, out_dir in zip(video_directories, output_dir):
        file_pattern = os.path.join(video_dir, "**", "*.jpg")
        video_files = glob.glob(file_pattern, recursive=True)
        frames.extend(video_files)
        frames = [cv2.imread(f) for f in frames]
        filename = os.path.basename(video_dir).split("/")[-1]
        filename = ".".join((filename, "jpg"))
        save_dynamic_image(frames, out_dir, filename)
        frames.clear()
    del input_dir, output_dir, video_directories


def save_dynamic_image(frames: list, output_directory: str, filename: str):
    """Save a dynamic image to the given directory.

    Args:
        frames (list): The list of frames to create the dynamic image.
        output_directory (str): The directory to save the dynamic image.
        filename (str): The name of the dynamic image.
    """
    dyn_image = get_dynamic_image(frames, normalized=True)
    output_path = os.path.join(output_directory, filename)

    if not os.path.exists(output_path):
        cv2.imwrite(output_path, dyn_image)
        # print('Saved dynamic image:', output_path)


def show_dynamic_image(frames):
    """Show a dynamic image.

    Args:
        frames (_type_): The list of frames to create the dynamic image.
    """
    dyn_image = get_dynamic_image(frames, normalized=True)
    cv2.imshow("", dyn_image)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            break
        if cv2.getWindowProperty("", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def get_dynamic_image(frames, normalized=True):
    """Takes a list of frames and returns either a raw or normalized dynamic image."""
    num_channels = frames[0].shape[2]
    channel_frames = get_channel_frames(frames, num_channels)
    channel_dynamic_images = [
        compute_dynamic_image(channel) for channel in channel_frames
    ]
    dynamic_image = cv2.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv2.normalize(
            dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX
        )
        dynamic_image = dynamic_image.astype("uint8")
    return dynamic_image


def get_channel_frames(iter_frames, num_channels):
    """Takes a list of frames and returns a list of frame lists split by channel."""
    frames = [[] for channel in range(num_channels)]
    for frame in iter_frames:
        for channel_frames, channel in zip(frames, cv2.split(frame)):
            channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
    for i in range(len(frames)):
        frames[i] = np.array(frames[i])
    return frames


def compute_dynamic_image(frames):
    """Adapted from https://github.com/hbilen/dynamic-image-nets"""
    num_frames, h, w, depth = frames.shape

    # Compute the coefficients for the frames.
    coefficients = np.zeros(num_frames)
    for n in range(num_frames):
        cumulative_indices = np.array(range(n, num_frames)) + 1
        coefficients[n] = np.sum(
            ((2 * cumulative_indices) - num_frames) / cumulative_indices
        )
    # Multiply by the frames by the coefficients and sum the result.
    x1 = np.expand_dims(frames, axis=0)
    x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
    result = x1 * x2
    return np.sum(result[0], axis=0).squeeze()


def extract_video_frames(self, video_dir: str, output_dir: str):
    """Extract the frames from a list of videos.

    Args:
        video_dir (str): The directory containing the videos.
        output_dir (str): The directory to save the frames.
    """
    seconds = []
    for video in video_dir:
        _, sec = self.get_video_duration(video)
        seconds.append(sec)
    max_duration = max(seconds)
    del seconds

    for video, out_dir in zip(video_dir, output_dir):
        filename = os.path.basename(video).split(".")[0]
        out_dir = "/".join((out_dir, filename))
        create_multiple_dirs(out_dir)

        vidcap = cv2.VideoCapture(video)
        frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
        extraction_rate = 0.5

        if max_duration > 10:
            extraction_rate = 1
        elif max_duration < 2:
            extraction_rate = 1 / max_duration

        success, image = vidcap.read()
        count = 0
        time_elapsed = 0

        while success:
            if time_elapsed >= extraction_rate:
                frame_path = os.path.join(out_dir, "frame%d.jpg" % count)

                if not os.path.exists(frame_path):
                    cv2.imwrite(frame_path, image)
                    # print('Saved frame:', frame_path)
                time_elapsed = 0

            success, image = vidcap.read()
            count += 1
            time_elapsed += 1 / frame_rate
        vidcap.release()
    del video_dir, output_dir


def get_video_duration(self, video_path: str):
    """Get the duration of a video.

    Args:
        video_path (str): The path to the video.

    Returns:
        _type_: The duration of the video.
    """
    data = cv2.VideoCapture(video_path)
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)
    seconds = round(frames / fps)
    video_time = datetime.timedelta(seconds=seconds)
    return video_time, seconds


def play_video(video_path: str):
    """Play a video from the given path.

    Args:
        video_path (str): The path to the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Frame", frame)
            if (
                cv2.waitKey(25) != -1
                or cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1
            ):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
