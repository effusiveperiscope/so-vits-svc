from einops import rearrange
import ultimate_xc
import os
from PyQt5.QtCore import pyqtRemoveInputHook
from PyQt5.QtWidgets import QApplication, QMainWindow
import librosa
import numpy as np
from logging import getLogger
from svc_helper.gui import *
from omegaconf import OmegaConf
from svc_helper.pitch.utils import nonzero_mean, discretize_f0_log, smooth_pitch
from svc_helper.speaker.models import SVC5SpeakerEncoder
from svc5whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
import sys
sys.path.append('..')
import torch
import soundfile as sf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from inference_util import InferTool

logger = getLogger(__name__)
CHECKPOINTS_ROOT = 'models'
CONFIG = 'configs/base.yaml'

class MainWindow(QMainWindow):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.setWindowTitle("so-vits-svc 5.0")
        self.setGeometry(100, 100, 800, 600)

        gui = VoiceGUI()
        gui.addCheckpoint(Checkpoint(
            get_checkpoints=self.getCheckpoints, load_checkpoint=self.loadCheckpoint))
        gui.addFileInput(AudioFileInput())
        gui.addFileInput(AudioFileInput(id='spk_files', label="Speaker Embedding Source"))
        gui.addParam(IntParam(label="Transpose", id='transpose', min=-24, max=24, default=0))
        gui.addParam(DoubleParam(label="Noise Scale", id='noise', min=0, max=3, default=0.5))
        gui.addInference(Inference(
            info=InferenceInfo(sr=32000, extension='flac'),
            infer_action=self.inferAction
        ))
        self.setCentralWidget(gui.build())

        self.infer_tool = InferTool()
        # We'll always do smoothing because RMVPE is our best available
        self.infer_tool.do_rmvpe_smoothing = True 

        self.dtype = torch.float32
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.svc5_spk_model = SVC5SpeakerEncoder(device=self.device)
        self.emb_file = None
        self.emb = None
        self.config = config


    def spkEmbMemoized(self, file : str):
        if file == self.emb_file:
            return self.emb
        self.emb_file = file
        with torch.no_grad():
            self.emb = self.svc5_spk_model.extract_feature(file).detach().cpu().numpy()
        return self.emb

    def getCheckpoints(self):
        return os.listdir(CHECKPOINTS_ROOT)

    def loadCheckpoint(self, checkpoint_name):
        logger.info(f'Loading checkpoint {checkpoint_name}')
        checkpoint_path = os.path.join(CHECKPOINTS_ROOT, checkpoint_name)
        true_checkpoint_path = next(
            (
                os.path.join(root, f)
                for root, _, files in os.walk(checkpoint_path)
                for f in files
                if f.endswith('.pt')
            ),
            None
        )
        spk_index = next(
            os.path.join(root, f)
            for root, _, files in os.walk(checkpoint_path)
            for f in files
            if f.endswith('.spk.npy')
        )
        # We will ignore cluster *.cluster 
        # and retrieval index checkpoints *hubert.index and *whisper.index
        # because cluster/retrieval index are not very useful

        self.infer_tool.load_svc_model(true_checkpoint_path)
        self.spk_index = self.infer_tool.load_speaker_emb(spk_index)
        self.cur_ckpt_path = true_checkpoint_path

        logger.info(f'Checkpoint {checkpoint_name} loaded')

    def inferAction(self, data: dict):
        transpose = data['transpose']
        files = data['audio_files']['files']

        if not hasattr(self, 'cur_ckpt_path'):
            logger.error('No checkpoint loaded')
            return InferenceResult(audios=[])

        logger.info(f'Inferring {len(files)} files')

        if len(data['audio_files']['spk_files']) > 0:
            if len(data['audio_files']['spk_files']) > 1:
                logger.warning('Only using first speaker embedding file')
            logger.info(f'Used speaker embedding from {data["audio_files"]["spk_files"][0]}')
            spk_files = data['audio_files']['spk_files']
            spk_feats = self.spkEmbMemoized(spk_files[0])
        else:
            spk_feats = None
            self.emb_file = None

        out = []
        for file in files:
            if spk_feats is None: # Fall back to index if none is provided
                spk_feats = self.spk_index

            with torch.no_grad():
                audio_data = load_audio(file)
                audio_data_32k = load_audio(file, sr=32000)
                o_np = self.infer_tool.infer(
                    audio_data=audio_data,
                    speaker_emb=spk_feats,
                    transpose=transpose,
                    f0_method='rmvpe',
                    x2_audio_data=audio_data_32k,
                    f0_mult_factor=1
                )
                out.append(AudioResult(
                    label=os.path.basename(file)+data['model_labels'][0],
                    audio=o_np))
        logger.info(f'Finished inferring {len(files)} files')
        return InferenceResult(audios=out)

if __name__ == '__main__':
    app = QApplication([])
    config = OmegaConf.load(CONFIG)
    window = MainWindow(config)
    window.show()
    app.exec_()