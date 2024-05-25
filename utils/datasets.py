import os
import csv
import pickle
import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import resample
import scipy.io as sio
import random
from tqdm import tqdm
import json
import wfdb

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def generate_split(X, Y, split, num_splits):
    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    for i in range(0, len(X)):
        if i % num_splits == split:
            X_test.append(X[i])
            Y_test.append(Y[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
    
    return np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test)

def ZhengEtAl(base_dir, shuffle=True, verbose=True, resample_rate=64):
    classes_of_interest = [ 270492004, # 1st degree AV block
                        284470004, # Atrial premature beats
                        426783006, # Sinus rhythm
                        164889003, # Atrial fibrillation
                        164890007, # Atrial fluttter
                        426761007  # Supraventricular tachycardia
                        ]

    class_names = ["1st degree AV block",
                "Atrial premature beats",
                #"Sinus bradycardia",
                "Sinus Rhythm",
                "Atrial fibrillation",
                #"Sinus tachycardia",
                "Atrial flutter",
                #"Sinus irregularity",
                "Supraventricular tachycardia",
                "Other"]
    
    with open(os.path.join(base_dir, "Zheng et al/ConditionNames_SNOMED-CT.csv"), "r") as f:
        condition_names = {}
        classes = []
        reader = csv.reader(f)
        for row in reader:
            condition_names[row[2]] = row[1]
        condition_names.pop('Snomed_CT')

    with open(os.path.join(base_dir, "Zheng et al/preprocessed_data.pkl"), "rb") as f:
        data = pickle.load(f)

    X = data["data"]
    X = list(X[:, 0, :])
    Y = data["labels"]
    classes = data["classes"]

    del data
    
    X_new = []
    Y_new = []

    # build each class
    for class_index, c in enumerate(classes_of_interest):
        for i in range(0, len(X)):
            if Y[i][classes.index(str(c))] == 1:
                add = True

                for c2 in classes_of_interest:
                    if c2 != c:
                        if Y[i][classes.index(str(c2))] == 1:
                            add = False
                
                if add:
                    X_new.append(X[i])
                    Y_new.append(class_index)

    # Build Other class
    for i in range(len(Y)):
        other = True
        for c in classes_of_interest:
            if Y[i][classes.index(str(c))] == 1.0:
                other = False
            if Y[i][classes.index('426177001')] == 1.0 or Y[i][classes.index('427084000')] == 1.0 or Y[i][classes.index('427393009')] == 1.0:
                other = False
        if other:
            X_new.append(X[i])
            Y_new.append(len(classes_of_interest))

    X = X_new
    Y = Y_new

    # get counts of each class
    if verbose:
        counts = np.zeros(len(classes_of_interest))
        for i in range(len(classes_of_interest)):
            counts[i] = np.sum(np.asarray(Y)==i)

        for i in range(len(classes_of_interest)):
            print(f"{condition_names[str(classes_of_interest[i])]}: {int(counts[i])}")

        print(f"Other: {int(np.sum(np.asarray(Y)==len(classes_of_interest)))}")

    for i in range(0, len(X)):
        # resample
        temp = np.float32(X[i])

        temp = resample(temp, len(temp) * resample_rate // 120)
        #temp = highpass_filter(temp)
        temp = bandpass_filter(temp, 1.0, 10.0, resample_rate)

        temp = (temp - np.mean(temp)) / np.std(temp)

        X[i] = np.float32(temp)
    
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = [X[i] for i in indices]
        Y = [Y[i] for i in indices]

    X = np.array(X)
    Y = np.array(Y)

    return X, Y, class_names


def ZhengEtAl_AVB(base_dir, shuffle=True, verbose=True):
    classes_of_interest = [ 270492004, # 1st degree AV block
                        426783006, # Sinus rhythm
                        ]

    class_names = ["1st degree AV block",
                "Sinus Rhythm",
                "Other"]
    
    with open(os.path.join(base_dir, "Zheng et al/ConditionNames_SNOMED-CT.csv"), "r") as f:
        condition_names = {}
        classes = []
        reader = csv.reader(f)
        for row in reader:
            condition_names[row[2]] = row[1]
        condition_names.pop('Snomed_CT')

    with open(os.path.join(base_dir, "Zheng et al/preprocessed_data.pkl"), "rb") as f:
        data = pickle.load(f)

    X = data["data"]
    X = list(X[:, 0, :])
    Y = data["labels"]
    classes = data["classes"]

    del data
    
    X_new = []
    Y_new = []

    # build each class
    for class_index, c in enumerate(classes_of_interest):
        for i in range(0, len(X)):
            if Y[i][classes.index(str(c))] == 1.0:
                add = True
                
                if add:
                    X_new.append(X[i])
                    Y_new.append(class_index)

    # Build Other class
    for i in range(len(Y)):
        other = True
        for c in classes_of_interest:
            if Y[i][classes.index(str(c))] == 1.0:
                other = False
            if Y[i][classes.index('426177001')] == 1.0 or Y[i][classes.index('427084000')] == 1.0 or Y[i][classes.index('427393009')] == 1.0:
                other = False
        if other:
            X_new.append(X[i])
            Y_new.append(len(classes_of_interest))

    X = X_new
    Y = Y_new

    # get counts of each class
    if verbose:
        counts = np.zeros(len(classes_of_interest))
        for i in range(len(classes_of_interest)):
            counts[i] = np.sum(np.asarray(Y)==i)

        for i in range(len(classes_of_interest)):
            print(f"{condition_names[str(classes_of_interest[i])]}: {int(counts[i])}")

        print(f"Other: {int(np.sum(np.asarray(Y)==len(classes_of_interest)))}")

    for i in range(0, len(X)):
        # resample
        temp = np.float32(X[i])

        temp = resample(temp, len(temp) * 64 // 120)
        temp = bandpass_filter(temp, 1.0, 10.0, 64)

        temp = (temp - np.mean(temp)) / np.std(temp)

        X[i] = np.float32(temp)
    
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = [X[i] for i in indices]
        Y = [Y[i] for i in indices]

    X = np.array(X)
    Y = np.array(Y)

    return X, Y, class_names

def chunk(x, WINDOW_SIZE=640):
    # number of windows
    num_windows = x.shape[-1] // WINDOW_SIZE

    # create windows tensor directly
    windows = x[:num_windows*WINDOW_SIZE].reshape((-1, WINDOW_SIZE))

    if x.shape[0] % WINDOW_SIZE != 0:
        # last window
        windows = np.concatenate((windows, x[-WINDOW_SIZE:].reshape(1, -1)), axis=0)

    return windows

def chunk_data(X, Y, chunk_size=640):
    new_X = []
    new_Y = []
    PowerSpectra = []

    for i, x in enumerate(X):
        if len(x) >= chunk_size:
            x_chunked = chunk(x, chunk_size)
            y = [Y[i]] * len(x_chunked)
            new_X.extend(list(x_chunked))
            new_Y.extend(y)
    
    return np.asarray(new_X), np.asarray(new_Y)

def load_from_dir(signal_dir, label_dir, exclude_fns=[], verbose=True):
        classes = ["N", "A", "O", "~"]

        # load labels
        with open(label_dir) as f:
            data = f.readlines()
            label_dict = {line.split(",")[0]: classes.index(line.split(",")[1].strip()) for line in data}

        train_files = []

        for file in os.listdir(signal_dir):
            if file.endswith(".mat"):
                train_files.append(file)
        
        X = []
        Y = []
        pbar = tqdm(train_files) if verbose else train_files
        for file in pbar:
            if file in exclude_fns:
                continue
            data = sio.loadmat(signal_dir + file)
            X.append(data['val'][0])
            Y.append(label_dict[file[:-4]])
        
        return X, Y, train_files

def CinC(base_dir, chunked=True, shuffle=True, verbose=True, resample_rate=64, chunk_size=640):
    # Load Data
    classes = ["N", "A", "O", "~"]

    X, Y, _ = load_from_dir(os.path.join(base_dir, "CinC/training2017/"), os.path.join(base_dir, "CinC/training2017/REFERENCE.csv"), verbose=verbose)

    # shuffle X and Y in unison
    c = list(zip(X, Y))
    random.shuffle(c)
    X, Y = zip(*c)
    X = list(X)
    Y = list(Y)

    Y = np.asarray(Y)

    for i in range(0, len(X)):
        # resample
        temp = np.float32(X[i])

        temp = resample(temp, len(temp) * resample_rate // 300)
        #temp = highpass_filter(temp)
        temp = bandpass_filter(temp, 1.0, 10.0, resample_rate)

        temp = (temp - np.mean(temp)) / np.std(temp)

        X[i] = np.float32(temp)

    if chunked:
        X, Y = chunk_data(X, Y, chunk_size)
    
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = [X[i] for i in indices]
        Y = [Y[i] for i in indices]
    
    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y, classes