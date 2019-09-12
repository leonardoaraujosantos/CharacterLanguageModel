from io import open
import unicodedata
import string
import numpy as np
import pickle

all_letters = string.ascii_letters + " .,;'-" + "0123456789"
n_letters = len(all_letters) + 1 # Plus EOS marker
set_classes = sorted(set(''.join(all_letters)))
EOS_token = n_letters
SOS_token = EOS_token+1

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Get Ascii code (Integer) from Unicode char
def getAsciiCode(s):
    return ord(unicodeToAscii(s))

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Get index for codemap(class) string from class integer
def class_id_from_char(char, codemap):
    #print(char)        
    if codemap is None:
        return 0
    else:   
        try:
            # Find key from value on dictionary
            return list(codemap.keys())[list(codemap.values()).index(char)]
        except:
            print('Class not found:', char)
            raise IndexError

def char_from_class_id(class_id, codemap):
    try:
        # Find key from value on dictionary
        char = codemap[class_id]
        if char == EOS_token:
            char = '<EOS>'
        if char == SOS_token:
            char = '<SOS>'
        return char
    except:
        print('Class not found:', class_id)
        raise IndexError

# Load codemap from pickle file            
def get_codemap_from_pickle(filename):
    try:
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)
        return b
    except FileNotFoundError:
        print('File not found')
        return None

# Load pickle file with strokes and labels
def load_pickle(data_file):
    try:
        f = open(data_file,"rb")
        dict_iam = pickle.load(f)
        f.close()
        return dict_iam
    except FileNotFoundError:
        return None
    
# Pad input and output sequence
def pad_data(lang_model_data, biggest_sequence):
    padded_X_class_idx = np.zeros((len(lang_model_data), biggest_sequence))    
    padded_Y_class_idx = np.zeros((len(lang_model_data), biggest_sequence))
    padded_Y_mask = np.zeros((len(lang_model_data), biggest_sequence))
    lang_model_data_pad = {}
    
    for idx, value in lang_model_data.items():
        #padded_X[i, 0:x_len] = sequence[:x_len]
        X_class_id, Y_class_id, len_x, len_y = value
        # Insert stroke values into padded_X
        padded_X_class_idx[idx, 0:len_x] = X_class_id[:len_x]
        # Insert stroke values into padded_Y    
        padded_Y_class_idx[idx, 0:len_y] = Y_class_id[:len_y]
        # Insert stroke ones into mask    
        padded_Y_mask[idx, 0:len_y] = 1
        
        lang_model_data_pad[idx] = padded_X_class_idx[idx, :], padded_Y_class_idx[idx, :], padded_Y_mask[idx, :], len_x, len_y
    
    return lang_model_data_pad