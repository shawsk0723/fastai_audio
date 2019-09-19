import sys
from pathlib import Path
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import torch
from fastai.vision import *
sys.path.append("..")
from audio import *

MODEL_PATH = "../data/ESC-50/models"
predictor = load_learner(MODEL_PATH)

def convertWav2AudioItem(sample_rate, signal):
    sg_cfg= SpectrogramConfig(hop=512, n_mels=128, n_fft=1024, top_db=80, f_min=20, f_max=22050)
    mel = MelSpectrogram(**(sg_cfg.mel_args()))(signal)
    mel = SpectrogramToDB(top_db=sg_cfg.top_db)(mel)
    mel = mel.permute(0, 2, 1)
    start, end = None, None
    return AudioItem(sig=signal.squeeze(), sr=sample_rate, spectro=mel, start=start, end=end)

def predict(sample_rate, data):
    audioItem = convertWav2AudioItem(sample_rate, data)
    return predictor.predict(audioItem)	

def test_predict():
    DATA_PATH = "../data/ESC-50/audio"
    POURING_WATER_FILE_NAME = "1-118559-A-17.wav"
    audio_file_path = DATA_PATH + '/' + POURING_WATER_FILE_NAME

    sig, sr = torchaudio.load(audio_file_path)
    print('sample rate = ', sr)
    print('sig = ', sig)

    predict_res = predict(sr, sig)
    print('predict result = ', predict_res)

if __name__ == '__main__':
    test_predict()
