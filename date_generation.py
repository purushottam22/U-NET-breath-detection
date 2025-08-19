import os
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import numpy as np
import pickle
from scipy.signal import decimate


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
    # Parameters

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


def get_st_end_time(file_name, label):
    ans, tmp_dic = [], {}
    tmp_dic["I"] = []
    tmp_dic["E"] = []
    inhale_count, exhale_count = 0, 0
    for i in range(len(label)):
        if label.iloc[i]["name"] == file_name:
            if label.iloc[i]["type"] == 'I':
                start, end = label.iloc[i]["start"], label.iloc[i]["end"]
                start = float(start.split(":")[-1])
                end = float(end.split(":")[-1])
                tmp_dic['I'].append([start, end])
                inhale_count+=1

            elif label.iloc[i]["type"] == 'E':
                start, end = label.iloc[i]["start"], label.iloc[i]["end"]
                start = float(start.split(":")[-1])
                end = float(end.split(":")[-1])
                tmp_dic['E'].append([start, end])
                exhale_count += 1

    if inhale_count == exhale_count:
        return tmp_dic
    else:
        return {}


def get_labels():
    label = pd.read_csv(r"breathing\label.csv")

    sound_dir = r"breathing\HF_Lung_V1-dataset\test"

    signal_output_path = r"output\filtered_500\3_split\signal"
    segment_label_output_path = r"output\filtered_500\3_split\segmented_label"

    names = label["name"]

    label_dic = []
    count, counti, counte = 0, 0, 0

    for j in tqdm(names):
        if j in label_dic:
            continue
        label_dic.append(j)
        labels_time = get_st_end_time(j, label)
        if len(labels_time) == 0:
            continue
        if len(labels_time["I"]) > 0 and len(labels_time["E"]) > 0:
            # print(j)
            counti += len(labels_time["I"])
            counte += len(labels_time["E"])
            count += 1
            name = "_".join(j.split("_")[0:-1])

            sound_file_name = name + ".wav"
            # Load the WAV file

            sound_path = os.path.join(sound_dir, sound_file_name)

            sample_rate, data = wavfile.read(sound_path)

            normalized_data = buttersmooth(data)

            segment_label = create_segmentaion(normalized_data, labels_time["I"], labels_time["E"])

            # Segment size
            segment_len = 3750  # 7.5 sec at 500 Hz
            stride = 1875  # 3.75 sec shift (50% overlap)

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


# get_labels()
