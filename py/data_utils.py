# data_utils.py
# Utility functions for data purposes
#
# author: Mark Thomas
# modfied: 2019-08-19

import os
import re
import shutil
import soundfile
import librosa
import torch
import random

import numpy as np

from torch.utils.data import Dataset

import warnings
warnings.filterwarnings('ignore')


class StreamingDataset(Dataset):
    """ Class for building the dataset to train MammalNet_v3.

    Attributes:
        identifiers (list): the string identifiers corresponding to the .wav files in the dataset
        file_dict (dict): a dictionary containing the metadate for all identifiers
        class_lookup (dict): a dictionary containing the class (str) and corresponding label (int)
        seconds (int): the number of seconds of the .wav file to use (default is 5s)
        fmax (int): the max frequency to consider (default is 1024Hz)
        local (bool): whether this is a local run or not (if so the directory of the .wav files is changed,
                      default is False)
        foreground (bool): whether to strip out the foreground from the spectrogram (default is False)
        ambient_prob (float): the probability that the example will be from the 'ambient' class (default is 0)
        pretraining (bool): whether this is a dataset being used for pretraining just the pretrain_backbone
                            ResNet model (default is False)
        transforms (callable, optional): optional transform to be applied to an instance
        test_seed (int): optional seed value to use during the sampling routine
        test_elsewhere (bool): whether this dataset is simply being used to get the time series
                               to compare to a different model/methodology (default is False)
        dtype (np.type): the datatype of the generated spectrograms
    """

    def __init__(self, identifiers, file_dict, class_lookup, seconds=5, fmax=1024, foreground=False,
                 ambient_prob=0, pretraining=False, local=False, transforms=None, test_seed=None, 
                 test_elsewhere=False, dtype=np.float32):
        self.IDENTIFIERS = identifiers
        self.FILE_DICT = file_dict
        self.CLASS_LOOKUP = class_lookup
        self.SECONDS = seconds
        self.FMAX = fmax
        self.FOREGROUND = foreground
        self.AMBIENT_PROB = ambient_prob
        self.PRETRAINING = pretraining
        self.LOCAL = local
        self.TRANSFORMS = transforms
        self.TEST_SEED = test_seed
        self.TEST_ELSEWHERE = test_elsewhere
        self.DTYPE = dtype

    def __len__(self):
        """ Returns the length of the list containing the wav files """
        return len(self.IDENTIFIERS)

    def __getitem__(self, idx):
        """ Creates a single instance of the dataset.

        Args:
            idx (int): the index of the .wav file that we want

        Returns:
            a single instance of the spectrograms
        """
        # Load the metadata corresponding to the id and 'idx'
        identifier = self.IDENTIFIERS[idx]

        # Sample the file based on the identifier
        ambient_example = True if np.random.randn() < self.AMBIENT_PROB else False 
        ts, sr, start_ind, end_ind = self.__read_wav_and_resample(identifier, ambient_example)
        
        # If we just want the time series to test elsewhere return now
        if self.TEST_ELSEWHERE:
            return ts, sr

        # Create the spectrograms
        features = self.__create_spectrogram_features(ts)
 
        # Create the sample dictionary
        if not self.PRETRAINING:
            assert self.AMBIENT_PROB == 0, 'Including only ambient files is only supported for pretraining ResNet'
            target = self.__build_bboxes(identifier, idx, features, start_ind, end_ind)
        else:
            if ambient_example:
                label = 0
            else:
                metadata = self.FILE_DICT[identifier]
                label = self.CLASS_LOOKUP[metadata['species']]
            target = torch.tensor(label, dtype=torch.int64)

        # Apply the transformations
        if self.TRANSFORMS is not None:
            features, target = self.TRANSFORMS(features, target)

        return features, target

    def __read_wav_and_resample(self, identifier, ambient_example, sr=8000):
        """ Loads a .wav file using librosa and then samples the waveform around a given annotation

        Args:
            identifier (str): the identifier containing all the relevant info for the file to be loaded
            sr (int): the sampling rate to by resampled to (default is 8kHz)

        Returns:
            the resampled time series (i.e., waveform) and the sampling rate
        """

        # Determine the file name
        metadata = self.FILE_DICT[identifier]
        filename = metadata['filepath']
        if self.LOCAL:
            filename = re.sub('/home/mdjt/scratch/JASCO/(.*)', '/Volumes/FAD062/all_esrf/full_files/\\1', filename)

        # Read the .wav file and resample it to match 'sr'
        ts, old_sr = soundfile.read(filename)
        ts = librosa.core.resample(ts, old_sr, sr)

        # Determine the start/end time of the annotation
        start = np.floor(metadata['starttime'])
        end = np.ceil(metadata['endtime'])
        duration = end - start

        # Find the sample indices
        if ambient_example:
            if start >= len(ts) // sr:
                start_ind = random.Random(self.TEST_SEED).randint(0, (len(ts) // sr) - self.SECONDS)
            elif np.floor(start - self.SECONDS) > 0:
                start_ind = int(np.floor(start - self.SECONDS))
            else:
                start_ind = int(np.ceil(end))
        else:
            if duration < self.SECONDS:
                under_duration = self.SECONDS - duration
                start_ind = random.Random(self.TEST_SEED).randint(start - under_duration, start)
            else:
                over_duration = duration - self.SECONDS
                start_ind = random.Random(self.TEST_SEED).randint(start, start + over_duration)

            # Make sure the start_ind falls within a proper range
            start_ind = np.maximum(0, start_ind)
            start_ind = np.minimum(int(len(ts) / 8000) - self.SECONDS, start_ind)

        # The end index is always the start + self.SECONDS
        end_ind = start_ind + self.SECONDS

        # Truncate the time series of the waveform
        ts = ts[start_ind*sr:end_ind*sr]
        
        return ts, sr, start_ind, end_ind

    def __create_spectrogram_features(self, data, sr=8000, nfft=2048):
        """ Generates a spectrogram using librosa.

        Args:
            data (np.array): the time series data of the waveform
            sr (int): the sampling rate of the data (default is 8000Hz)
            nfft (int): the length of the FFT (default is 2048 samples)

        Returns:
            a numpy array containing the STFT matrix
        """
        # Generate the spectrogram
        D = librosa.core.stft(data, n_fft=nfft, dtype=self.DTYPE)

        # Remove phase from the matrix D
        D, _ = librosa.magphase(D)

        # Strip out the foreground
        if self.FOREGROUND:
            D_filter = librosa.decompose.nn_filter(D, aggregate=np.median, metric='cosine', width=1)
            D_filter = np.minimum(D, D_filter)
            mask = librosa.util.softmask(D - D_filter, 10 * D_filter, power=2)
            D = mask * D

        # Convert to log scale i.e., decibels (dB)
        D = librosa.amplitude_to_db(D, ref=np.max)

        # Truncate the spectrogram to 1000Hz
        f_temp = librosa.fft_frequencies(sr=8000, n_fft=2048)
        f_keep = np.where(f_temp <= self.FMAX)
        D = D[f_keep, :][0, :, :]

        # Reshape the numpy array to have channels
        D = np.reshape(D, (1, D.shape[0], D.shape[1]))

        # Rescale the numpy array between 0 and 1
        features = (D - D.min()) / (D.max() - D.min())
        features = torch.as_tensor(features, dtype=torch.float32)

        return features

    def __build_bboxes(self, identifier, idx, features, start_ind, end_ind):
        """ Builds the bounding boxe target for a given identifier.

        Args:
            identifier (str): the identifier of the annotation
            idx (int): the index of the requested __getitem__
            features (np.array): the matrix of features (i.e., spectrogram)
            start_ind (int): the start index of the time series
            end_ind (int): the end index of the time series

        Returns:
            a dictionary containing the target information
        """

        # Get the metadata/label corresponding to the supplied identifier
        metadata = self.FILE_DICT[identifier]
        label = self.CLASS_LOOKUP[metadata['species']]

        # Get the time indices of the first bounding box relative to the feature matrix
        times = librosa.core.times_like(features, sr=8000, hop_length=512, n_fft=2048)
        start_time = np.maximum(start_ind, metadata['starttime']) - start_ind
        end_time = np.minimum(end_ind, metadata['endtime']) - start_ind
        start_time_ind = np.maximum(np.argmin(times <= start_time) - 1, 0)
        end_time_ind = np.minimum(np.argmax(times >= end_time) + 1, features.shape[2])

        # Get the frequency indices of the first bounding box relative to the feature matrix
        freqs = librosa.core.fft_frequencies(sr=8000, n_fft=2048)
        low_freq = metadata['lowfreq']
        high_freq = metadata['highfreq']
        low_freq_ind = np.maximum(np.argmin(freqs <= low_freq) - 1, 0)
        high_freq_ind = np.minimum(np.argmax(freqs >= high_freq) + 1, features.shape[1])

        # Initialize the first bounding box
        labels = [label]
        boxes = [[start_time_ind, low_freq_ind, end_time_ind, high_freq_ind]]

        # Add the overlap identifiers to the bounding boxes/labels
        overlap_ids = metadata['overlap_identifiers'].split(', ')
        if overlap_ids[0]:
            # If there are more than 5 just take a sample of only 5 of them
            if len(overlap_ids) > 5:
                overlap_ids = random.Random(self.TEST_SEED).sample(overlap_ids, 5)

            # Loop through the overlap identifiers and add those that are still within frame
            for id in overlap_ids:
                # Get the metadata/label corresponding to the overlapped identifier
                tmp_metadata = self.FILE_DICT[id]
                tmp_label = self.CLASS_LOOKUP[tmp_metadata['species']]

                # Determine the start/end times for the overlapped identifier
                tmp_start_time = tmp_metadata['starttime']
                tmp_end_time = tmp_metadata['endtime']

                # Make a bunch of conditions that need to be satisfied for this bounding box to be included
                cond_1 = (tmp_start_time <= end_ind) and (tmp_end_time <= end_ind) and (tmp_start_time >= start_ind) and (tmp_end_time >= start_ind)
                cond_2 = (tmp_start_time <= end_ind) and (tmp_end_time <= end_ind) and (tmp_start_time <= start_ind) and (tmp_end_time >= start_ind)
                cond_3 = (tmp_start_time <= end_ind) and (tmp_end_time >= end_ind) and (tmp_start_time >= start_ind) and (tmp_end_time >= start_ind)
                cond_4 = (tmp_start_time <= end_ind) and (tmp_end_time >= end_ind) and (tmp_start_time <= start_ind) and (tmp_end_time >= start_ind)

                # If all the conditions pass then add the bounding box
                if cond_1 or cond_2 or cond_3 or cond_4:
                    # Truncate the start/end times to be within the visible part of the time series
                    tmp_start_time = np.maximum(tmp_start_time, start_ind) - start_ind
                    tmp_end_time = np.minimum(tmp_end_time, end_ind) - start_ind

                    # Get the time indices of the first bounding box relative to the feature matrix
                    tmp_start_time_ind = np.maximum(np.argmin(times <= tmp_start_time) - 1, 0)
                    tmp_end_time_ind = np.minimum(np.argmax(times >= tmp_end_time) + 1, features.shape[2])

                    # Get the frequency indices of the overlapped bounding box relative to the feature matrix
                    tmp_low_freq = tmp_metadata['lowfreq']
                    tmp_high_freq = tmp_metadata['highfreq']
                    tmp_low_freq_ind = np.maximum(np.argmin(freqs <= tmp_low_freq) - 1, 0)
                    tmp_high_freq_ind = np.minimum(np.argmax(freqs >= tmp_high_freq) + 1, features.shape[1])

                    # Make the bounding box for this overlapped identifier
                    tmp_box = [tmp_start_time_ind, tmp_low_freq_ind, tmp_end_time_ind, tmp_high_freq_ind]

                    # Append the box and label
                    boxes.append(tmp_box)
                    labels.append(tmp_label)

        # Convert the arrays to Tensors and compute the areas
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes), ), dtype=torch.int64)

        # Make the target dict and return
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['id'] = torch.tensor([idx])
        target['area'] = areas
        target['iscrowd'] = iscrowd

        return target


class StreamFullRecording(Dataset):
    """ Class for building the dataset to test MammalNet_v3 on a full recording

    Attributes:
        filename (str): the file to read
        annotations (dict): the annotations for a given file
        class_lookup (dict): a dictionary containing the class (str) and corresponding label (int)
        seconds (int): the number of seconds of the .wav file to use (default is 5)
        stepsize (int): the number of frames to overlap (default it 400 - roughly 20fps)
        fmax (int): the max frequency to consider (default is 512Hz)
        local (bool): whether this is a local run or not (default is False)
        foreground (bool): whether to strip out the foreground from the spectrogram (default is False)
        transforms (callable, optional): optional transform to be applied to an instance
        dtype (np.type): the datatype of the generated spectrograms
    """

    def __init__(self, filename, annotations, class_lookup, seconds=5, stepsize=400, fmax=512,
                 foreground=False, local=False, transforms=None, dtype=np.float32):
        self.FILENAME = filename
        self.ANNOTATIONS = annotations
        self.CLASS_LOOKUP = class_lookup
        self.SECONDS = seconds
        self.STEPSIZE = stepsize
        self.FMAX = fmax
        self.FOREGROUND = foreground
        self.LOCAL = local
        self.TRANSFORMS = transforms
        self.DTYPE = dtype

    def __len__(self):
        """ Returns the length of dataset
        """
        ts, sr = self.__read_wav_and_resample(self.FILENAME)

        length, start = (0, 0)
        end = start + (self.SECONDS * sr)
        while end < len(ts):
            length += 1
            start += self.STEPSIZE
            end = start + (self.SECONDS * sr)

        return(length)

    def __getitem__(self, idx):
        """ Creates a single instance of the dataset.

        Args:
            idx (int): the index of the .wav file that we want

        Returns:
            a single instance of the spectrograms
        """
        # Sample the file based on the identifier
        ts, sr = self.__read_wav_and_resample(self.FILENAME)

        # Strip out the parts of the time series according to the idx requested
        start_frame = idx * self.STEPSIZE
        end_frame = start_frame + (self.SECONDS * sr)
        ts = ts[start_frame:end_frame]

        # Compute the features and build the bounding boxes
        features = self.__create_spectrogram_features(ts)
        targets = self.__build_bboxes(idx, features, start_frame, end_frame, sr)

        # Apply the transformations
        if self.TRANSFORMS is not None:
            features, targets = self.TRANSFORMS(features, targets)

        return features, targets

    def __read_wav_and_resample(self, filename, sr=8000):
        """ Loads a .wav file using librosa and then samples the waveform around a given annotation

        Args:
            filename (str): the filename to read
            sr (int): the sampling rate to by resampled to (default is 8kHz)

        Returns:
            the resampled time series (i.e., waveform) and the sampling rate
        """
        # If running locally, then change the filename to match
        if self.LOCAL:
            filename = re.sub('/home/mdjt/scratch/JASCO/(.*)', '/Volumes/FAD062/all_esrf/full_files/\\1', filename)

        # Read the .wav file and resample it to match 'sr'
        ts, old_sr = soundfile.read(filename)
        ts = librosa.core.resample(ts, old_sr, sr)

        return ts, sr

    def __create_spectrogram_features(self, data, sr=8000, nfft=2048):
        """ Generates a spectrogram using librosa.

        Args:
            data (np.array): the time series data of the waveform
            sr (int): the sampling rate of the data (default is 8000Hz)
            nfft (int): the length of the FFT (default is 2048 samples)

        Returns:
            a numpy array containing the STFT matrix
        """
        # Generate the spectrogram
        D = librosa.core.stft(data, n_fft=nfft, dtype=self.DTYPE)

        # Remove phase from the matrix D
        D, _ = librosa.magphase(D)

        # Strip out the foreground
        if self.FOREGROUND:
            D_filter = librosa.decompose.nn_filter(D, aggregate=np.median, metric='cosine', width=1)
            D_filter = np.minimum(D, D_filter)
            mask = librosa.util.softmask(D - D_filter, 10 * D_filter, power=2)
            D = mask * D

        # Convert to log scale i.e., decibels (dB)
        D = librosa.amplitude_to_db(D, ref=np.max)

        # Truncate the spectrogram to 1000Hz
        f_temp = librosa.fft_frequencies(sr=8000, n_fft=2048)
        f_keep = np.where(f_temp <= self.FMAX)
        D = D[f_keep, :][0, :, :]

        # Reshape the numpy array to have channels
        D = np.reshape(D, (1, *D.shape))

        # Rescale the numpy array between 0 and 1
        features = (D - D.min()) / (D.max() - D.min())
        features = torch.as_tensor(features, dtype=torch.float32)

        return features

    def __build_bboxes(self, idx, features, start_frame, end_frame, sr):
        """ Builds the bounding boxe target for a given identifier.

        Args:
            idx (int): the index of the requested __getitem__
            features (np.array): the matrix of features (i.e., spectrogram)
            start_frame (int): the start frame of the time series
            end_frame (int): the end frame of the time series
            sr (int): the sampling rate of the time series

        Returns:
            a dictionary containing the target information
        """
        def filter_conds(ann_start, ann_end, ts_start, ts_end):
            ''' Checks four conditions to ensure the annotation should be included

            Args:
                ann_start (float): the start time in seconds of the annotation
                ann_end (float): the start time in seconds of the annotation
                ts_start (float): the start time in seconds of the time series
                                  corresponding to the input features
                ts_end (float): the start time in seconds of the time series
                                corresponding to the input features

            Returns:
                True or False (whether any of the conditions were met)
            '''
            cond_1 = (ann_start <= ts_end) and (ann_end <= ts_end) and (ann_start >= ts_start) and (ann_end >= ts_start)
            cond_2 = (ann_start <= ts_end) and (ann_end <= ts_end) and (ann_start <= ts_start) and (ann_end >= ts_start)
            cond_3 = (ann_start <= ts_end) and (ann_end >= ts_end) and (ann_start >= ts_start) and (ann_end >= ts_start)
            cond_4 = (ann_start <= ts_end) and (ann_end >= ts_end) and (ann_start <= ts_start) and (ann_end >= ts_start)
            return cond_1 or cond_2 or cond_3 or cond_4

        # Change the start/end inputs to seconds instead of frames
        start_sec = start_frame / sr
        end_sec = end_frame / sr

        # Filter the annotations that are visible and create the time/freq reference arrays
        annotations = {k: v for (k, v) in self.ANNOTATIONS.items() if filter_conds(v['starttime'], v['endtime'], start_sec, end_sec)}
        times = librosa.core.times_like(features, sr=8000, hop_length=512, n_fft=2048)
        freqs = librosa.core.fft_frequencies(sr=8000, n_fft=2048)

        # Initialize the bounding box and labels
        labels = []
        boxes = []

        # Loop through the annotations
        for annotation in annotations.values():
            # Get the time indices of the first bounding box relative to the feature matrix
            tmp_start = np.maximum(annotation['starttime'], start_sec) - start_sec
            temp_end = np.minimum(annotation['endtime'], end_sec) - start_sec
            start_ind = np.maximum(np.argmin(times <= tmp_start) - 1, 0)
            end_ind = np.minimum(np.argmax(times >= temp_end) + 1, features.shape[2])

            # Get the frequency indices of the overlapped bounding box relative to the feature matrix
            low_freq_ind = np.maximum(np.argmin(freqs <= annotation['lowfreq']) - 1, 0)
            high_freq_ind = np.minimum(np.argmax(freqs >= annotation['highfreq']) + 1, features.shape[1])

            # Append the box and label
            boxes.append([start_ind, low_freq_ind, end_ind, high_freq_ind])
            labels.append(self.CLASS_LOOKUP[annotation['species']])

        # Convert the arrays to Tensors and compute the areas
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if len(boxes) > 0:
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            areas = torch.zeros((0, ), dtype=torch.float32)

        # Suppose all instances are not "crowded"
        iscrowd = torch.zeros((len(boxes), ), dtype=torch.int64)

        # Make the target dict and return
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['id'] = torch.tensor([idx])
        target['area'] = areas
        target['iscrowd'] = iscrowd

        return target


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    """ Saves the model as the current checkpoint, either in general
        or as the current most best, accordingly.

        Args:
            state (dict): the current state of the model
            is_best (bool): whether this is the new best run
            filename (str): the name of the checkpoint file (default is
                            checkpoint.pth.tar)
    """
    model_dir = './models/{}'.format(args.run_id)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    with open('{}/{}'.format(model_dir, filename), 'wb') as file:
        torch.save(state, file)

    if is_best:
        shutil.copyfile('{}/{}'.format(model_dir, filename), '{}/model_best.pth.tar'.format(model_dir))
