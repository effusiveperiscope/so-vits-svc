import os
import torch
import numpy as np
import time
import pyworld

from omegaconf import OmegaConf
from scipy.io.wavfile import write
import scipy.signal as signal
from vits.models import SynthesizerInfer
from pitch import load_csv_pitch

from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
from hubert import hubert_model
from sklearn.cluster import KMeans
from pathlib import Path
from feature_retrieval import IRetrieval, DummyRetrieval, FaissIndexRetrieval, load_retrieve_index

LOG_TIMES = True
RMVPE_PATH = Path("rmvpe.pt")

class InferTool:
    def __init__(self,
        whisper_path = os.path.join("whisper_pretrain","large-v2.pt"),
        hubert_path = os.path.join("hubert_pretrain","hubert-soft-0d54a1f4.pt"),
        config_dir = "configs/base.yaml"):
        self.hp = OmegaConf.load(config_dir)
        self.model = None
        self.whisper = self.load_whisper_model(whisper_path)
        self.hubert = self.load_hubert_model(hubert_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retrieval = None
        pass

    def load_retrieval(self,
        hubert_path, whisper_path,
        ratio, n_nearest_vectors) -> IRetrieval:
        if (hubert_path is None) or (whisper_path is None):
            self.retrieval = None
            return self.retrieval
        #print(f"Loading retrieval models from {hubert_path}, {whisper_path}")
        self.retrieval = FaissIndexRetrieval(
            hubert_index=load_retrieve_index(
                filepath=hubert_path,
                ratio=ratio,
                n_nearest_vectors=n_nearest_vectors
            ),
            whisper_index=load_retrieve_index(
                filepath=whisper_path,
                ratio=ratio,
                n_nearest_vectors=n_nearest_vectors
            ),
        )
        return self.retrieval

    def set_retrieval_ratio(self, ratio: float):
        if self.retrieval is None:
            return
        self.retrieval._ratio = ratio

    def set_retrieval_n_vec(self, n_nearest_vectors: int):
        if self.retrieval is None:
            return
        self.retrieval._n_nearest = n_nearest_vectors

    def load_whisper_model(self, path) -> Whisper:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=device)
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device)

    def load_hubert_model(self, path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = hubert_model.hubert_soft(path)
        model.eval()
        model.half()
        model.to(device)
        return model

    def load_svc_model(self, checkpoint_path):
        assert os.path.isfile(checkpoint_path)

        model = SynthesizerInfer(
            self.hp.data.filter_length // 2 + 1,
            self.hp.data.segment_size // self.hp.data.hop_length,
            self.hp)

        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
        saved_state_dict = checkpoint_dict["model_g"]
        state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            try:
                new_state_dict[k] = saved_state_dict[k]
            except:
                print("%s is not in the checkpoint" % k)
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.eval()
        model.to(self.device)

        self.model = model
        # print("loaded model from "+checkpoint_path)
        pass

    def pred_vec(self, wav_data):
        audio = wav_data
        audln = audio.shape[0]
        vec_a = []
        idx_s = 0
        while (idx_s + 20 * 16000 < audln):
            feats = audio[idx_s:idx_s + 20 * 16000]
            feats = torch.from_numpy(feats).to(self.device)
            feats = feats[None, None, :].half()
            with torch.no_grad():
                vec = self.hubert.units(
                    feats).squeeze().data.cpu().float().numpy()
                vec_a.extend(vec)
            idx_s = idx_s + 20 * 16000
        if (idx_s < audln):
            feats = audio[idx_s:audln]
            feats = torch.from_numpy(feats).to(self.device)
            feats = feats[None, None, :].half()
            with torch.no_grad():
                vec = self.hubert.units(
                    feats).squeeze().data.cpu().float().numpy()
                # print(vec.shape)   # [length, dim=256] hop=320
                vec_a.extend(vec)
        return vec_a

    def pred_ppg(self, wav_data):
        audio = wav_data 
        audln = audio.shape[0]
        ppg_a = []
        idx_s = 0
        while (idx_s + 25 * 16000 < audln):
            short = audio[idx_s:idx_s + 25 * 16000]
            idx_s = idx_s + 25 * 16000
            ppgln = 25 * 16000 // 320
            # short = pad_or_trim(short)
            mel = log_mel_spectrogram(short).to(self.device)
            with torch.no_grad():
                ppg = self.whisper.encoder(
                    mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
                ppg = ppg[:ppgln,]  # [length, dim=1024]
                ppg_a.extend(ppg)
        if (idx_s < audln):
            short = audio[idx_s:audln]
            ppgln = (audln - idx_s) // 320
            mel = log_mel_spectrogram(short).to(self.device)
            with torch.no_grad():
                ppg = self.whisper.encoder(
                    mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
                ppg = ppg[:ppgln,]  # [length, dim=1024]
                ppg_a.extend(ppg)
        return ppg_a

    def compute_f0_harvest(self, audio):
        import pyworld
        hop_length = 320
        sr = 16000
        p_len = audio.shape[0]//hop_length
        f0, t = pyworld.harvest(
            audio.astype(np.double),
            fs=sr,
            f0_ceil=800,
            frame_period=1000 * hop_length / sr,
        )
        f0 = pyworld.stonemask(audio.astype(np.double), f0, t, sr)
        #for index, pitch in enumerate(f0): # This doesn't make any sense...
            #f0[index] = round(pitch, 1) # ???
        f0 = np.repeat(f0, 2, -1)
        return f0

    def compute_f0_harvest2(self, audio):
        import pyworld
        hop_length = 320
        sr = 32000
        p_len = audio.shape[0]//hop_length
        f0, t = pyworld.harvest(
            audio.astype(np.double),
            fs=sr,
            f0_ceil=800,
            frame_period=1000 * hop_length / sr,
        ) # this is half of the size collected for the thing, not double.
        f0 = pyworld.stonemask(audio.astype(np.double), f0, t, sr)
        #for index, pitch in enumerate(f0): # This doesn't make any sense...
            #f0[index] = round(pitch, 1) # ???
        f0 = signal.medfilt(f0, 3)
        return f0

    def compute_f0_parselmouth_cc(self, audio, voice_thresh = 0.3):
        import parselmouth
        sr = 16000
        hop_length = 320
        time_step = hop_length / sampling_rate * 1000
        f0 = parselmouth.Sound(audio, sr).to_pitch_cc(
            time_step / 1000, voicing_threshold = voice_thresh,
            pitch_floor = 75, pitch_ceiling = 1100).selected_array['frequency']
        f0 = np.repeat(f0, 2, -1)
        return f0

    def compute_f0_nn(self, audio):
        import torchcrepe
        # Load audio
        audio = torch.tensor(np.copy(audio))[None]
        hop_length = 320
        fmin = 50
        fmax = 1000
        model = "full"
        batch_size = 512
        sr = 16000
        pitch, periodicity = torchcrepe.predict(
            audio,
            sr,
            hop_length,
            fmin,
            fmax,
            model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
        pitch = torchcrepe.filter.mean(pitch, 3)
        pitch[periodicity < 0.05] = 0
        pitch = pitch.squeeze(0)
        return pitch

    def compute_f0_nn2(self, audio):
        import torchcrepe
        # Load audio
        audio = torch.tensor(np.copy(audio))[None]
        hop_length = 320
        fmin = 50
        fmax = 1000
        model = "full"
        batch_size = 512
        sr = 32000
        pitch, periodicity = torchcrepe.predict(
            audio,
            sr,
            hop_length,
            fmin,
            fmax,
            model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        periodicity = torchcrepe.filter.median(periodicity, 9)
        pitch = torchcrepe.filter.mean(pitch, 3)
        pitch[periodicity < 0.05] = 0
        pitch = pitch.squeeze(0)
        return pitch

    def compute_f0_rmvpe(self, audio):
        if (hasattr(self, "rmvpe") == False) and RMVPE_PATH.is_file():
            from rmvpe import RMVPE
            self.rmvpe = RMVPE(
                RMVPE_PATH, is_half=False, device=self.device
            )
        f0 = self.rmvpe.infer_from_audio(audio, thred=0.03)
        return f0

    def load_speaker_emb(self, speaker_emb_file):
        return np.load(speaker_emb_file)

    def load_cluster(self, cluster_path):
        if cluster_path is not None:
            ckpt = torch.load(cluster_path)
            self.cluster_model = KMeans(ckpt["n_features_in_"])
            self.cluster_model.__dict__["n_features_in_"] = (
                ckpt["n_features_in_"])
            self.cluster_model.__dict__["_n_threads"] = (
                ckpt["_n_threads"])
            self.cluster_model.__dict__["cluster_centers_"] = (
                ckpt["cluster_centers_"])
        else:
            self.cluster_model = None

    # It is assumed input data is mono 16bit pcm 16k sr.
    # We can use load_audio from whisper/audio.py to comply with this
    def infer(self,
        audio_data : np.ndarray,
        speaker_emb : np.ndarray,
        transpose = 0,
        f0_method = "harvest2",
        x2_audio_data = None):

        # Hypothesis: Higher resolution audio data is needed
        # for pitch detection.
        assert self.model is not None

        start_time = time.time()

        ppg = np.array(self.pred_ppg(audio_data))
        vec = np.array(self.pred_vec(audio_data))

        ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
        ppg = torch.FloatTensor(ppg)

        vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
        vec = torch.FloatTensor(vec)

        ppg_time = time.time()
        if LOG_TIMES:
            print("ppg time: "+str(ppg_time - start_time))

        if f0_method == "harvest":
            pit = self.compute_f0_harvest(audio_data)
        if f0_method == "harvest2":
            pit = self.compute_f0_harvest2(x2_audio_data)
        elif f0_method == "crepe":
            pit = self.compute_f0_nn(audio_data)
        elif f0_method == "crepe2":
            pit = self.compute_f0_nn2(x2_audio_data)
        elif f0_method == "parselmouth":
            pit = self.compute_f0_parselmouth_cc(audio_data)
        elif f0_method == "rmvpe":
            pit = self.compute_f0_rmvpe(audio_data)
        print(pit.shape)
        print(pit.mean())
        pit = pit * (2 ** (transpose / 12))
        pit = torch.FloatTensor(pit)

        pit_time = time.time()
        if LOG_TIMES:
            print("pit time: "+str(pit_time - ppg_time))

        len_pit = pit.size()[0]
        len_vec = vec.size()[0]
        len_ppg = ppg.size()[0]
        len_min = min(len_pit, len_ppg)
        len_min = min(len_min, len_vec)
        pit = pit[:len_min]
        vec = vec[:len_min, :]
        ppg = ppg[:len_min, :]

        spk = speaker_emb
        spk = torch.FloatTensor(spk)

        with torch.no_grad():
            spk = spk.unsqueeze(0).to(self.device)
            source = pit.unsqueeze(0).to(self.device)
            source = self.model.pitch2source(source)

            hop_size = self.hp.data.hop_length
            all_frame = len_min
            hop_frame = 10
            out_chunk = 2500  # 25 S
            out_index = 0
            out_audio = []

            while (out_index < all_frame):
                if (out_index == 0):  # start frame
                    cut_s = 0
                    cut_s_out = 0
                else:
                    cut_s = out_index - hop_frame
                    cut_s_out = hop_frame * hop_size

                # end frame
                if (out_index + out_chunk + hop_frame > all_frame):  
                    cut_e = all_frame
                    cut_e_out = -1
                else:
                    cut_e = out_index + out_chunk + hop_frame
                    cut_e_out = -1 * hop_frame * hop_size

                if self.retrieval is not None:
                    print("Using retrieval")
                    sub_ppg = self.retrieval.retriv_whisper(
                        ppg[cut_s:cut_e, :]).unsqueeze(0).to(self.device)
                    sub_vec = self.retrieval.retriv_hubert(
                        vec[cut_s:cut_e, :]).unsqueeze(0).to(self.device)
                else:
                    sub_ppg = ppg[cut_s:cut_e, :].unsqueeze(0).to(self.device)
                    sub_vec = vec[cut_s:cut_e, :].unsqueeze(0).to(self.device)
                sub_pit = pit[cut_s:cut_e].unsqueeze(0).to(self.device)
                sub_len = torch.LongTensor([cut_e - cut_s]).to(self.device)
                sub_har = source[:, :, cut_s *
                                 hop_size:cut_e * hop_size].to(self.device)
                sub_out = self.model.inference(
                    sub_ppg, sub_vec, sub_pit, spk, sub_len, sub_har)
                sub_out = sub_out[0, 0].data.cpu().detach().numpy()

                sub_out = sub_out[cut_s_out:cut_e_out]
                out_audio.extend(sub_out)
                out_index = out_index + out_chunk

            out_audio = np.asarray(out_audio)

        infer_time = time.time()
        if LOG_TIMES:
            print("infer time: "+str(infer_time - pit_time))

        return out_audio
