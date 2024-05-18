import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import librosa
import torch
from rmvpe import RMVPE
import argparse
from tqdm import tqdm

# Download the checkpoint at https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt

def longpath(path):
    import platform
    path = os.path.abspath(path)
    if 'Windows' in platform.system() and not path.startswith('\\\\?\\'):
        path = u'\\\\?\\'+path.replace('/','\\')
        return path
    else:
        return path

RMVPE_MODEL_PATH = 'rmvpe.pt'
rmvpe = RMVPE(RMVPE_MODEL_PATH, is_half=False, device=device)

def compute_f0(filename, save, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    # F0
    pitch = rmvpe.infer_from_audio(audio, thred=1e-3)
    np.save(save, pitch, allow_pickle=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-p", "--pit", help="pit", dest="pit", required=True)

    args = parser.parse_args()
    print(args.wav)
    print(args.pit)

    os.makedirs(args.pit, exist_ok=True)
    wavPath = args.wav
    pitPath = args.pit

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{pitPath}/{spks}", exist_ok=True)

            files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]
            for file in tqdm(files, desc=f'Processing rmvpe {spks}'):
                file = file[:-4]
                compute_f0(
                    longpath(f"{wavPath}/{spks}/{file}.wav"), 
                    longpath(f"{pitPath}/{spks}/{file}.pit"), device)
