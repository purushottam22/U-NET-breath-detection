import os
import numpy as np
import pickle
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, decimate
from tqdm import tqdm


def downsample(signal):
    downsampled_signal = decimate(signal, 8, ftype='iir', zero_phase=True)
    return downsampled_signal


def min_max_normalization(data):
    min_val = min(data)
    max_val = max(data)

    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]

    return normalized_data


def buttersmooth(signal):
    data = downsample(signal)
    sample_rate = 500
    order = 10
    cutoff_freq = 80  # in Hz
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist  # normalize frequency

    # Design Butterworth high-pass filter
    b, a = butter(order, normalized_cutoff, 'high', analog=True)

    # Apply the filter to the signal
    filtered_data = filtfilt(b, a, data)

    normalized_data = min_max_normalization(filtered_data)
    return normalized_data


def create_segmentaion(data, inhalation_times, exhalation_times, sample_rate=500):
    total_samples = len(data)

    # 1D target label array initialized to 0 (background)
    target = np.zeros(total_samples, dtype=np.uint8)

    # Function to mark a time segment in the label array
    def mark_segment(label_array, segments, label, sr):
        for start_sec, end_sec in segments:
            start_idx = int(start_sec * sr)
            end_idx = int(end_sec * sr)
            label_array[start_idx:end_idx] = label

    # Mark inhalation (1) and exhalation (2)
    mark_segment(target, inhalation_times, label=1, sr=sample_rate)
    mark_segment(target, exhalation_times, label=2, sr=sample_rate)

    return target


def parse_time(time_str):
    """Convert HH:MM:SS.mmm timestamp to total seconds."""
    parts = time_str.split(":")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_label_file(label_path):
    """
    Parse a per-recording label TXT file with format: TYPE START END
    Keeps only I (inhalation) and E (exhalation) lines.
    Skips D (duplicate marker) and Rhonchi (adventitious sound) lines.
    Returns dict: {'I': [[start, end], ...], 'E': [[start, end], ...]}
    """
    inhalations = []
    exhalations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            label_type, start_str, end_str = parts
            if label_type == 'I':
                inhalations.append([parse_time(start_str), parse_time(end_str)])
            elif label_type == 'E':
                exhalations.append([parse_time(start_str), parse_time(end_str)])
    return {'I': inhalations, 'E': exhalations}


def process_directory(sound_dir, signal_output_path, segment_label_output_path):
    """Process all WAV files in a directory, saving preprocessed segments."""
    wav_files = sorted([f for f in os.listdir(sound_dir) if f.endswith('.wav')])

    for wav_file in tqdm(wav_files, desc=f"Processing {sound_dir}"):
        name = wav_file[:-4]  # strip .wav
        label_path = os.path.join(sound_dir, name + "_label.txt")

        if not os.path.exists(label_path):
            continue

        labels = parse_label_file(label_path)

        if len(labels['I']) == 0 or len(labels['E']) == 0:
            continue

        sound_path = os.path.join(sound_dir, wav_file)
        sample_rate, data = wavfile.read(sound_path)

        normalized_data = buttersmooth(data)
        segment_label = create_segmentaion(normalized_data, labels['I'], labels['E'])

        # Segment size
        segment_len = 3750  # 7.5 sec at 500 Hz
        stride = 1875       # 3.75 sec shift (50% overlap)

        # Create and save three segments
        for idx in range(3):
            start = idx * stride
            end = start + segment_len

            if end <= len(normalized_data):
                seg_data = normalized_data[start:end]
                seg_label = segment_label[start:end]

                file_prefix = name + f"_part{idx + 1}"

                with open(os.path.join(signal_output_path, file_prefix + "_data.pkl"), 'wb') as f1:
                    pickle.dump(seg_data, f1)

                with open(os.path.join(segment_label_output_path, file_prefix + "_label.pkl"), 'wb') as f2:
                    pickle.dump(seg_label, f2)


def get_labels():
    signal_output_path = r"output\filtered_500\3_split\signal"
    segment_label_output_path = r"output\filtered_500\3_split\segmented_label"

    os.makedirs(signal_output_path, exist_ok=True)
    os.makedirs(segment_label_output_path, exist_ok=True)

    for split in ['train', 'test']:
        if os.path.isdir(split):
            process_directory(split, signal_output_path, segment_label_output_path)


if __name__ == "__main__":
    get_labels()
