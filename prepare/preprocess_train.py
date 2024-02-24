import os
import random
import argparse

def longpath(path):
    import platform
    path = os.path.abspath(path)
    if 'Windows' in platform.system() and not path.startswith('\\\\?\\'):
        path = u'\\\\?\\'+path.replace('/','\\')
        return path
    else:
        return path

def print_error(info):
    print(f"\033[31m File isn't existed: {info}\033[0m")


IndexBySinger = False
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data_dir", help="data_svc directory")
    parser.add_argument("-r","--raw_dir", help="dataset_raw directory")
    args = parser.parse_args()
    os.makedirs("./files/", exist_ok=True)

    rawPath = "./"+args.raw_dir
    dataPath = "./"+args.data_dir
    all_items = []
    for spks in os.listdir(f"./{rawPath}"):
        if not os.path.isdir(f"./{rawPath}/{spks}"):
            continue
        print(f"./{rawPath}/{spks}")
        for file in os.listdir(f"./{rawPath}/{spks}"):
            if file.endswith(".wav"):
                file = file[:-4]
                path_spk = longpath(dataPath+f"/speaker/{spks}/{file}.spk.npy")
                path_wave = longpath(dataPath+f"/waves-32k/{spks}/{file}.wav")
                path_spec = longpath(dataPath+f"/specs/{spks}/{file}.pt")
                path_pitch = longpath(dataPath+f"/pitch/{spks}/{file}.pit.npy")
                path_hubert = longpath(dataPath+f"/hubert/{spks}/{file}.vec.npy")
                path_whisper = longpath(dataPath+f"/whisper/{spks}/{file}.ppg.npy")
                has_error = 0
                if not os.path.isfile(path_spk):
                    print_error(path_spk)
                    has_error = 1
                if not os.path.isfile(path_wave):
                    print_error(path_wave)
                    has_error = 1
                if not os.path.isfile(path_spec):
                    print_error(path_spec)
                    has_error = 1
                if not os.path.isfile(path_pitch):
                    print_error(path_pitch)
                    has_error = 1
                if not os.path.isfile(path_hubert):
                    print_error(path_hubert)
                    has_error = 1
                if not os.path.isfile(path_whisper):
                    print_error(path_whisper)
                    has_error = 1
                if has_error == 0:
                    all_items.append(
                        f"{path_wave}|{path_spec}|{path_pitch}|{path_hubert}|{path_whisper}|{path_spk}")

    random.shuffle(all_items)
    valids = all_items[:1]
    valids.sort()
    trains = all_items[1:]
    # trains.sort()
    fw = open("./files/valid.txt", "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()
    fw = open("./files/train.txt", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()
