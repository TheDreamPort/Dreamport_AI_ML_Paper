import torch
import sys
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from attrib import DatasetAttributes
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import files as f
import os
from joblib import dump, load
import copy

def load_train_and_val_data(file_name, train_attrib=DatasetAttributes()):
    '''Loads data as numpy arrays and converts them to tensors.
    Returns TensorDatasets containing inputs and outputs for validation and training.'''
    inputs, outputs, train_attrib = load_numpy_data(
        file_name, train_attrib)
    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(inputs, outputs, train_size = 0.96)
    val_attrib = copy.deepcopy(train_attrib)
    val_attrib.num_examples = val_outputs.shape[0]
    train_attrib.num_examples = train_outputs.shape[0]
    return make_tensor_dataset(train_inputs, train_outputs), make_tensor_dataset(val_inputs, val_outputs), train_attrib, val_attrib

def load_test_data(file_name, attrib):
    '''Loads data as numpy arrays and converts them to tensors.
    Returns TensorDataset containing inputs and outputs for testing.'''
    inputs, outputs, attrib = load_numpy_data(
        file_name, attrib)
    return make_tensor_dataset(inputs, outputs), attrib

def make_tensor_dataset(inputs, outputs):
    tensor_inputs = torch.tensor(inputs, dtype=torch.float)
    tensor_outputs = torch.tensor(outputs, dtype=torch.long)
    return TensorDataset(tensor_inputs, tensor_outputs)

def load_numpy_data(file_name, attrib=DatasetAttributes()):
    '''Checks whether the data has already been preprocessed. Preprocesses if
    necessary.'''
    in_file = file_name + '_in.npy'
    out_file = file_name + '_out.npy'
    if os.path.isfile(f.ATTRIB):
        attrib = load(f.ATTRIB)
    if os.path.isfile(in_file) and os.path.isfile(out_file):
        return load_preprocessed_data(in_file, out_file, attrib)
    else:
        return preprocess_data(file_name, in_file, out_file, attrib)


def load_preprocessed_data(in_file, out_file, attrib):
    '''Loads preprocessed numpy arrays.'''
    try:
        inputs = np.load(in_file)
        outputs = np.load(out_file)
    except Exception as e:
        print("Could not open data file." + str(e))
        sys.exit()
    attrib.feature_length = inputs.shape[1]
    attrib.num_examples = inputs.shape[0]
    return inputs, outputs, attrib

def load_text(file_name):
    try:
        data = np.loadtxt(open(file_name + '.txt', "rb"),
                          delimiter=",", dtype="S")
    except Exception as e:
        print("Could not open data file. " + str(e))
        sys.exit()
    return data

def preprocess_data(file_name, in_file, out_file, attrib=DatasetAttributes()):
    '''Loads a .csv file of PCAP data, splits it into inputs and outputs, and
    normalizes it. The second-to-last column is the labels of the rows, such as
    "normal" or "neptune". The last column is ignored. Rows labelled "normal"
    are labelled as 0 (normal) in the outputs, and all other rows are labelled 1
    (malicious). Saves the preprocessed data so that this only needs to be done
    once.
    '''
    data = load_text(file_name)
    find_str_types(data[:, :-2], attrib)
    inputs = convert_to_nums(data[:, :-2], attrib.str_inputs)
    outputs = convert_to_nums_binary(data[:, -2])
    inputs = inputs.astype(np.float)
    outputs = outputs.astype(np.long)
    scaler = RobustScaler()
    inputs = scaler.fit(inputs).transform(inputs)
    attrib.feature_length = inputs.shape[1]
    attrib.num_examples = inputs.shape[0]
    try:
        np.save(in_file, inputs)
        np.save(out_file, outputs)
        dump(attrib, f.ATTRIB)
    except Exception as e:
        print("Could not save preprocessed data. " + str(e))
    return inputs, outputs, attrib


def convert_to_nums(data, str_types):
    '''Converts a string array with mixed numbers and strings to one
    with only numbers. Columns that used to have strings end up zeroed, with
    the strings converted to one-hot categories at the end of the vectors.'''
    num_data = np.zeros((data.shape[0], data.shape[1] + len(str_types)), dtype=np.float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            cell = data[i, j]
            if not cell.replace(b'.', b'', 1).isdigit():
                num_data[i, data.shape[1] + str_types.index(cell)] = 1
            else:
                num_data[i, j] = float(cell)
    return num_data

def find_str_types(data, attrib):
    for i, cell in np.ndenumerate(data):
        if not cell.replace(b'.', b'', 1).isdigit():
            if not cell in attrib.str_inputs:
                attrib.str_inputs.append(cell)

def convert_to_nums_binary(data):
    '''Replaces labels with 0 for normal and 1 for everything else.'''
    for i, cell in np.ndenumerate(data):
        if cell == b"normal":
            data[i] = b'0'
        else:
            data[i] = b'1'
    return data


def build_dataloaders():
    '''Loads data from files into batched tensor dataloaders.'''
    train_dataset, val_dataset, train_attrib, val_attrib = load_train_and_val_data(
        f.TRAIN)
    test_dataset, test_attrib = load_test_data(
        f.TEST, train_attrib)
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
    }
    attrib_dict = {
        'train': train_attrib,
        'val': val_attrib,
        'test': test_attrib,
    }
    dataloaders = {x: DataLoader(datasets[x], batch_size=1024,
                                 num_workers=4, shuffle=True,
                                 pin_memory=torch.cuda.is_available()
                                 ) for x in ['train', 'val', 'test']}
    return dataloaders, attrib_dict

def load_str_labels():
    '''Find the names of the attack types and normal records.'''
    all_train_labels = load_text(f.TRAIN)[:, -2]
    all_test_labels = load_text(f.TEST)[:, -2]
    unique_train_labels = np.unique(all_train_labels)
    unique_test_labels = np.unique(all_test_labels)
    print(all_test_labels)
    d = {
        'novel': unique_test_labels[np.isin(unique_test_labels, unique_train_labels, invert=True)],
        'shared': unique_test_labels[np.isin(unique_test_labels, unique_train_labels)]
        }
    str_labels = {}
    for t in ['novel', 'shared']:
        str_labels[t] = {}
        for label in d[t]:
            str_labels[t][label] = {
                'total': np.where(all_test_labels == label)[0].shape[0],
                'correct': 0,
                }
    return str_labels, all_test_labels


if __name__ == '__main__':
    # Just preprocess, save the data, build the dataloaders, and throw them out.
    # Used to profile loading speed
    build_dataloaders()
