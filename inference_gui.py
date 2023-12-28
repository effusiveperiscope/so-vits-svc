import argparse
import math
import traceback
import io
import os
import logging
import time
import sys
import copy
import importlib.util
from ctypes import cast, POINTER, c_int, c_short, c_float
from pathlib import Path
from PyQt5.QtCore import (pyqtSignal, Qt, QUrl, QSize, QMimeData, QMetaObject,
    pyqtSlot)
from PyQt5.QtGui import (QIntValidator, QDoubleValidator, QKeySequence,
    QDrag)
from PyQt5.QtMultimedia import (
   QMediaContent, QAudio, QAudioDeviceInfo, QMediaPlayer, QAudioRecorder,
   QAudioEncoderSettings, QMultimedia,
   QAudioProbe, QAudioFormat)
from PyQt5.QtWidgets import (QWidget,
   QSizePolicy, QStyle, QProgressBar,
   QApplication, QMainWindow,
   QFrame, QFileDialog, QLineEdit, QSlider,
   QPushButton, QHBoxLayout, QVBoxLayout, QLabel,
   QPlainTextEdit, QComboBox, QGroupBox, QCheckBox, QShortcut, QDialog)
import numpy as np
from scipy.io.wavfile import write
import glob
import json
import torch
import subprocess
from datetime import datetime
from collections import deque
from pathlib import Path
import librosa
from inference_util import InferTool
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram

RECORD_SHORTCUT = "ctrl+shift+r"
if (importlib.util.find_spec("pygame")):
    from pygame import mixer, _sdl2 as devicer
    import pygame._sdl2.audio as sdl2_audio
    print("Automatic mode enabled. Press "+RECORD_SHORTCUT+
        " to toggle recording.")
    PYGAME_AVAILABLE = True
else:
    print("Note: Automatic mode not available."
    "To enable: pip install pygame keyboard")
    PYGAME_AVAILABLE = False

if importlib.util.find_spec("requests"):
    import requests
    REQUESTS_AVAILABLE = True
else:
    REQUESTS_AVAILABLE = False

if importlib.util.find_spec("pedalboard"):
    import pedalboard
    PEDALBOARD_AVAILABLE = True
else:
    PEDALBOARD_AVAILABLE = False

if (subprocess.run(["where","rubberband"] if os.name == "nt" else 
    ["which","rubberband"]).returncode == 0) and importlib.util.find_spec(
        "pyrubberband"):
    #print("Rubberband is available!")
    import pyrubberband as pyrb
    RUBBERBAND_AVAILABLE = True
else:
    #print("Note: Rubberband is not available. Timestretch not available.")
    RUBBERBAND_AVAILABLE = False

TALKNET_ADDR = "127.0.0.1:8050"
MODELS_DIR = "models"
BASE_CFG_DIR = "configs/base.yaml"
RECORD_DIR = "./recordings"
JSON_NAME = "inference_gui_persist.json"
RECENT_DIR_MAXLEN = 10
F0_OPTIONS = ["rmvpe", "crepe2", "harvest2", "crepe", "harvest", "parselmouth"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def get_speakers():
    speakers = []
    for folder in os.listdir(MODELS_DIR):
        cur_speaker = {}

        # Look for *.pt
        g = glob.glob(os.path.join(MODELS_DIR,folder,"*.pt"))
        if not len(g):
            print("Skipping "+folder+", no *.pt")
            continue
        cur_speaker["model_path"] = g[0]
        cur_speaker["model_folder"] = folder

        # Look for singer
        if not os.path.exists(os.path.join(MODELS_DIR,folder,"singer")):
            print("Skipping "+folder+", no singer folder")
            continue
        g = glob.glob(os.path.join(MODELS_DIR,folder,"singer","*.spk.npy"))
        for name in g:
            cur_speaker["name"] = Path(name).stem.split('.')[0]
            spk_name = cur_speaker["name"]
            cur_speaker["spk_path"] = name

            # Look for cluster checkpoints
            cluster_path = os.path.join(MODELS_DIR,folder,"cluster",
                cur_speaker["name"]+".cluster")
            if os.path.exists(cluster_path):
                cur_speaker["cluster_path"] = cluster_path
            else:
                cur_speaker["cluster_path"] = None

            # Look for retrieval indexes
            hubert_g = glob.glob(
                os.path.join(
                    MODELS_DIR,folder,"indexes",spk_name,"*hubert.index"))
            whisper_g = glob.glob(
                os.path.join(
                    MODELS_DIR,folder,"indexes",spk_name,"*whisper.index"))
            #print(hubert_g)
            if (len(hubert_g) and len(whisper_g)):
                cur_speaker["index_paths"] = {
                    "hubert": hubert_g[0], "whisper": whisper_g[0]}
            else:
                cur_speaker["index_paths"] = {
                    "hubert": None,
                    "whisper": None
                }

            speakers.append(copy.copy(cur_speaker))

    return sorted(speakers, key=lambda x:x["name"].lower())

def el_trunc(s, n=80):
    return s[:min(len(s),n-3)]+'...'

def backtruncate_path(path, n=80):
    if len(path) < (n):
        return path
    path = path.replace('\\','/')
    spl = path.split('/')
    pth = spl[-1]
    i = -1

    while len(pth) < (n - 3):
        i -= 1
        if abs(i) > len(spl):
            break
        pth = os.path.join(spl[i],pth)

    spl = pth.split(os.path.sep)
    pth = os.path.join(*spl)
    return '...'+pth

logging.getLogger('numba').setLevel(logging.WARNING)

os.makedirs("raw",exist_ok=True)
os.makedirs("results",exist_ok=True)
wav_format = 'wav'

class FieldWidget(QFrame):
    def __init__(self, label, field):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0,0,0,0)
        label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(label)
        self.field = field
        field.setAlignment(Qt.AlignRight)
        field.sizeHint = lambda: QSize(60, 32)
        field.setSizePolicy(QSizePolicy.Maximum,
            QSizePolicy.Preferred)
        self.layout.addWidget(field)

    def setEnabled(self, val):
        self.field.setEnabled(val)

class VSTWidget(QWidget):
    sig_editor_open = pyqtSignal(bool)
    def __init__(self):
        # this should not even be loaded if pedalboard is not available
        assert PEDALBOARD_AVAILABLE 
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.select_button = QPushButton("No VST loaded")
        self.select_button.setSizePolicy(QSizePolicy.Expanding,
            QSizePolicy.Preferred)
        self.select_button.clicked.connect(self.select_plugin)
        self.editor_button = QPushButton("Open UI")
        self.editor_button.clicked.connect(self.open_editor)
        self.layout.addWidget(self.select_button)
        self.layout.addWidget(self.editor_button)
        self.bypass_button = QCheckBox("Bypass")
        self.layout.addWidget(self.bypass_button)
        self.plugin_container = None

    def select_plugin(self):
        file = QFileDialog.getOpenFileName(self, "Plugin to load")
        if not len(file[0]):
            return
        try:
            self.plugin_container = pedalboard.VST3Plugin(file[0])
            self.select_button.setText(self.plugin_container.name)
        except ImportError as e:
            self.plugin_container = None
            self.select_button.setText("No VST loaded")

    def open_editor(self):
        if self.plugin_container is not None:
            self.sig_editor_open.emit(True)
            self.plugin_container.show_editor()
            self.sig_editor_open.emit(False)

    def process(self, array, sr):
        if self.plugin_container is None:
            return array
        if self.bypass_button.isChecked():
            return array
        return self.plugin_container.process(
            input_array = np.array(array),
            sample_rate = float(sr))

class AudioPreviewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.vlayout = QVBoxLayout(self)
        self.vlayout.setSpacing(0)
        self.vlayout.setContentsMargins(0,0,0,0)

        self.playing_label = QLabel("Preview")
        self.playing_label.setWordWrap(True)
        self.vlayout.addWidget(self.playing_label)

        self.player_frame = QFrame()
        self.vlayout.addWidget(self.player_frame)

        self.player_layout = QHBoxLayout(self.player_frame)
        self.player_layout.setSpacing(4)
        self.player_layout.setContentsMargins(0,0,0,0)

        #self.playing_label.hide()

        self.player = QMediaPlayer()
        self.player.setNotifyInterval(500)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setSizePolicy(QSizePolicy.Expanding,
            QSizePolicy.Preferred)
        self.player_layout.addWidget(self.seek_slider)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(
            getattr(QStyle, 'SP_MediaPlay')))
        self.player_layout.addWidget(self.play_button)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setSizePolicy(QSizePolicy.Maximum,
            QSizePolicy.Minimum)
        self.play_button.mouseMoveEvent = self.drag_hook

        self.seek_slider.sliderMoved.connect(self.seek)
        self.player.positionChanged.connect(self.update_seek_slider)
        self.player.stateChanged.connect(self.state_changed)
        self.player.durationChanged.connect(self.duration_changed)

        self.local_file = ""

    def set_text(self, text=""):
        if len(text) > 0:
            self.playing_label.show()
            self.playing_label.setText(text)
        else:
            self.playing_label.hide()

    def from_file(self, path):
        try:
            self.player.stop()
            if hasattr(self, 'audio_buffer'):
                self.audio_buffer.close()

            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(
                os.path.abspath(path))))

            self.play_button.setIcon(self.style().standardIcon(
                getattr(QStyle, 'SP_MediaPlay')))

            self.local_file = path
        except Exception as e:
            pass

    def drag_hook(self, e):
        if e.buttons() != Qt.LeftButton:
            return
        if not len(self.local_file):
            return

        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(
            os.path.abspath(self.local_file))])
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec_(Qt.CopyAction)

    def from_memory(self, data):
        self.player.stop()
        if hasattr(self, 'audio_buffer'):
            self.audio_buffer.close()

        self.audio_data = QByteArray(data)
        self.audio_buffer = QBuffer()
        self.audio_buffer.setData(self.audio_data)
        self.audio_buffer.open(QBuffer.ReadOnly)
        player.setMedia(QMediaContent(), self.audio_buffer)

    def state_changed(self, state):
        if (state == QMediaPlayer.StoppedState) or (
            state == QMediaPlayer.PausedState):
            self.play_button.setIcon(self.style().standardIcon(
                getattr(QStyle, 'SP_MediaPlay')))

    def duration_changed(self, dur):
        self.seek_slider.setRange(0, self.player.duration())

    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        elif self.player.mediaStatus() != QMediaPlayer.NoMedia:
            self.player.play()
            self.play_button.setIcon(self.style().standardIcon(
                getattr(QStyle, 'SP_MediaPause')))

    def update_seek_slider(self, position):
        self.seek_slider.setValue(position)

    def seek(self, position):
        self.player.setPosition(position)

class AudioRecorderAndVSTs(QGroupBox):
    keyboardRecordSignal = pyqtSignal()
    def __init__(self, par):
        super().__init__()
        self.setTitle("Audio recorder and VST processing")
        self.setStyleSheet("padding:10px")
        self.layout = QVBoxLayout(self)
        self.ui_parent = par

        self.audio_settings = QAudioEncoderSettings()
        if os.name == "nt":
            self.audio_settings.setCodec("audio/pcm")
        else:
            self.audio_settings.setCodec("audio/x-raw")
        self.audio_settings.setSampleRate(44100)
        self.audio_settings.setBitRate(16)
        self.audio_settings.setQuality(QMultimedia.HighQuality)
        self.audio_settings.setEncodingMode(
            QMultimedia.ConstantQualityEncoding)

        self.preview = AudioPreviewWidget()
        self.layout.addWidget(self.preview)

        self.recorder = QAudioRecorder()
        self.input_dev_box = QComboBox()
        self.input_dev_box.setSizePolicy(QSizePolicy.Preferred,
            QSizePolicy.Preferred)
        if os.name == "nt":
            self.audio_inputs = self.recorder.audioInputs()
        else:
            self.audio_inputs = [x.deviceName() 
                for x in QAudioDeviceInfo.availableDevices(0)]

        self.record_button = QPushButton("Record")
        self.record_button.clicked.connect(self.toggle_record)
        self.layout.addWidget(self.record_button)

        for inp in self.audio_inputs:
            if self.input_dev_box.findText(el_trunc(inp,60)) == -1:
                self.input_dev_box.addItem(el_trunc(inp,60))
        self.layout.addWidget(self.input_dev_box)
        self.input_dev_box.currentIndexChanged.connect(self.set_input_dev)
        if len(self.audio_inputs) == 0:
            self.record_button.setEnabled(False) 
            print("No audio inputs found")
        else:
            self.set_input_dev(0) # Always use the first listed output
        # Doing otherwise on Windows would require platform-specific code

        if PYGAME_AVAILABLE and importlib.util.find_spec("keyboard"):
            try:
                print("Keyboard module loaded.")
                print("Recording shortcut without window focus enabled.")
                import keyboard
                def keyboard_record_hook():
                    self.keyboardRecordSignal.emit()
                keyboard.add_hotkey(RECORD_SHORTCUT,keyboard_record_hook)
                self.keyboardRecordSignal.connect(self.toggle_record)
            except ImportError as e:
                print("Keyboard module failed to import.")
                print("On Linux, must be run as root for recording"
                    "hotkey out of focus.")
                self.record_shortcut = QShortcut(QKeySequence(RECORD_SHORTCUT),
                    self)
                self.record_shortcut.activated.connect(self.toggle_record)
        else:
            print("No keyboard module available.")
            print("Using default input capture for recording shortcut.")
            self.record_shortcut = QShortcut(QKeySequence(RECORD_SHORTCUT),
                self)
            self.record_shortcut.activated.connect(self.toggle_record)

        self.probe = QAudioProbe()
        self.probe.setSource(self.recorder)
        self.probe.audioBufferProbed.connect(self.update_volume)
        self.volume_meter = QProgressBar()
        self.volume_meter.setTextVisible(False)
        self.volume_meter.setRange(0, 100)
        self.volume_meter.setValue(0)
        self.layout.addWidget(self.volume_meter)

        if PYGAME_AVAILABLE:
            self.record_out_label = QLabel("Output device")
            mixer.init()
            self.out_devs = sdl2_audio.get_audio_device_names(False)
            mixer.quit()
            self.output_dev_box = QComboBox()
            self.output_dev_box.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Preferred)
            for dev in self.out_devs:
                if self.output_dev_box.findText(el_trunc(dev,60)) == -1:
                    self.output_dev_box.addItem(el_trunc(dev,60))
            self.output_dev_box.currentIndexChanged.connect(self.set_output_dev)
            self.selected_dev = None
            self.set_output_dev(0)
            self.layout.addWidget(self.record_out_label)
            self.layout.addWidget(self.output_dev_box)

        # RECORD_DIR
        self.record_dir = os.path.abspath(RECORD_DIR)
        self.record_dir_button = QPushButton("Change Recording Directory")
        self.layout.addWidget(self.record_dir_button)
        self.record_dir_label = QLabel("Recordings directory: "+str(
            self.record_dir))
        self.record_dir_button.clicked.connect(self.record_dir_dialog)

        self.last_output = ""

        self.sovits_button = QPushButton("Push last output to so-vits-svc")
        self.layout.addWidget(self.sovits_button)
        self.sovits_button.clicked.connect(self.push_to_sovits)

        self.automatic_checkbox = QCheckBox("Send automatically")
        self.layout.addWidget(self.automatic_checkbox)

        if PYGAME_AVAILABLE:
            self.mic_checkbox = QCheckBox(
                "Auto-play output to selected output device")
            self.layout.addWidget(self.mic_checkbox)
            self.mic_checkbox.stateChanged.connect(self.update_init_audio)

            self.mic_output_control = QCheckBox("Auto-delete audio from "
                "recordings/results after auto-playing")
            self.layout.addWidget(self.mic_output_control)
            self.mic_output_control.stateChanged.connect(self.update_delfiles)
        
        if (par.talknet_available):
            self.talknet_button = QPushButton("Push last output to TalkNet")
            self.layout.addWidget(self.talknet_button)
            self.talknet_button.clicked.connect(self.push_to_talknet)

        if PEDALBOARD_AVAILABLE:
            self.vst_input_frame = QGroupBox(self)
            self.vst_input_frame.setTitle("so-vits-svc Pre VSTs")
            self.vst_input_layout = QVBoxLayout(self.vst_input_frame)
            self.layout.addWidget(self.vst_input_frame)
            self.vst_inputs = []
            for i in range(2):
                vst_widget = VSTWidget()
                self.vst_inputs.append(vst_widget)
                self.vst_input_layout.addWidget(vst_widget)
                vst_widget.sig_editor_open.connect(
                    self.ui_parent.pass_editor_ctl)

            self.vst_output_frame = QGroupBox(self)
            self.vst_output_frame.setTitle("so-vits-svc Post VSTs")
            self.vst_output_layout = QVBoxLayout(self.vst_output_frame)
            self.layout.addWidget(self.vst_output_frame)
            self.vst_outputs = []
            for i in range(2):
                vst_widget = VSTWidget()
                self.vst_outputs.append(vst_widget)
                self.vst_output_layout.addWidget(vst_widget)
                vst_widget.sig_editor_open.connect(
                    self.ui_parent.pass_editor_ctl)
        
        self.layout.addStretch()

    def output_chain(self, data, sr):
        if PEDALBOARD_AVAILABLE:
            for v in self.vst_outputs:
                data = v.process(data, sr)
        return data

    def input_chain(self, data, sr):
        if PEDALBOARD_AVAILABLE:
            for v in self.vst_inputs:
                data = v.process(data, sr)
        return data

    def update_volume(self, buf):
        sample_size = buf.format().sampleSize()
        sample_count = buf.sampleCount()
        ptr = buf.constData()
        ptr.setsize(int(sample_size/8)*sample_count)

        samples = np.asarray(np.frombuffer(ptr, np.int16)).astype(float)
        rms = np.sqrt(np.mean(samples**2))
            
        level = rms / (2 ** 14)

        self.volume_meter.setValue(int(level * 100))

    def update_init_audio(self):
        if PYGAME_AVAILABLE:
            mixer.init(devicename = self.selected_dev)
            if self.mic_checkbox.isChecked():
                self.ui_parent.mic_state = True
            else:
                self.ui_parent.mic_state = False

    def update_delfiles(self):
        self.ui_parent.mic_delfiles = self.mic_output_control.isChecked()

    def set_input_dev(self, idx):
        num_audio_inputs = len(self.audio_inputs)
        if idx < num_audio_inputs:
            self.recorder.setAudioInput(self.audio_inputs[idx])

    def set_output_dev(self, idx):
        self.selected_dev = self.out_devs[idx]
        if mixer.get_init() is not None:
            mixer.quit()
            mixer.init(devicename = self.selected_dev)

    def record_dir_dialog(self):
        temp_record_dir = QFileDialog.getExistingDirectory(self,
            "Recordings Directory", self.record_dir, QFileDialog.ShowDirsOnly)
        if not os.path.exists(temp_record_dir): 
            return
        self.record_dir = temp_record_dir
        self.record_dir_label.setText(
            "Recordings directory: "+str(self.record_dir))
        
    def toggle_record(self):
        #print("toggle_record triggered at "+str(id(self)))
        if self.recorder.status() == QAudioRecorder.RecordingStatus:
            self.recorder.stop()
            self.record_button.setText("Record")
            self.last_output = self.recorder.outputLocation().toLocalFile()
            if not (PYGAME_AVAILABLE and self.mic_output_control.isChecked()):
                self.preview.from_file(self.last_output)
                self.preview.set_text("Preview - "+os.path.basename(
                    self.last_output))
            if self.automatic_checkbox.isChecked():
                self.push_to_sovits()
                self.ui_parent.sofvits_convert()
        else:
            self.record()
            self.record_button.setText("Recording to "+str(
                self.recorder.outputLocation().toLocalFile()))

    def record(self):
        unix_time = time.time()
        self.recorder.setEncodingSettings(self.audio_settings)
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir, exist_ok=True)
        output_name = "rec_"+str(int(unix_time))
        self.recorder.setOutputLocation(QUrl.fromLocalFile(os.path.join(
            self.record_dir,output_name)))
        self.recorder.setContainerFormat("audio/x-wav")
        self.recorder.record()

    def push_to_sovits(self):
        if not os.path.exists(self.last_output):
            return
        self.ui_parent.clean_files = [self.last_output]
        self.ui_parent.update_file_label()
        self.ui_parent.update_input_preview()

    def push_to_talknet(self):
        if not os.path.exists(self.last_output):
            return
        self.ui_parent.talknet_file = self.last_output
        self.ui_parent.talknet_file_label.setText(
            "File: "+str(self.ui_parent.talknet_file))
        self.ui_parent.talknet_update_preview()

class FileButton(QPushButton):
    fileDropped = pyqtSignal(list)
    def __init__(self, label = "Files to Convert"):
        super().__init__(label)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            clean_files = []
            for url in event.mimeData().urls():
                if not url.toLocalFile():
                    continue
                clean_files.append(url.toLocalFile())
            self.fileDropped.emit(clean_files)
            event.acceptProposedAction()
        else:
            event.ignore()
        pass

class InferenceGui2 (QMainWindow):
    def __init__(self, args):
        super().__init__()

        self.mic_state = False
        self.mic_delfiles = False
        self.clean_files = [0]
        self.infer_tool = InferTool()
        self.speakers = get_speakers()
        self.speaker = {}
        self.output_dir = os.path.abspath("./results/")
        self.cached_file_dir = os.path.abspath(".")
        self.recent_dirs = deque(maxlen=RECENT_DIR_MAXLEN)

        self.setWindowTitle("so-vits-svc 5.0 GUI")
        self.central_widget = QFrame()
        self.layout = QHBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        self.sovits_frame = QGroupBox(self)
        self.sovits_frame.setTitle("so-vits-svc")
        self.sovits_frame.setStyleSheet("padding:10px")
        self.sovits_lay = QVBoxLayout(self.sovits_frame)
        self.sovits_lay.setSpacing(0)
        self.sovits_lay.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.sovits_frame)

        self.load_persist()
        self.talknet_available = self.try_connect_talknet()

        # Cull non-existent paths from recent_dirs
        self.recent_dirs = deque(
            [d for d in self.recent_dirs if os.path.exists(d)], maxlen=RECENT_DIR_MAXLEN)
        
        self.speaker_box = QComboBox()
        for spk in self.speakers:
            self.speaker_box.addItem(spk["name"]+" ["+
                Path(spk["model_folder"]).stem+"]")
        self.speaker_frame = QFrame()
        self.speaker_frame_layout = QHBoxLayout(self.speaker_frame)
        self.speaker_label = QLabel("Speaker:")
        self.speaker_label.setWordWrap(True)
        self.speaker_frame_layout.addWidget(self.speaker_label)

        self.sovits_lay.addWidget(self.speaker_frame)
        self.sovits_lay.addWidget(self.speaker_box)
        self.speaker_box.currentIndexChanged.connect(self.try_load_speaker)

        self.file_button = FileButton()
        self.sovits_lay.addWidget(self.file_button)
        self.file_label = QLabel("Files: "+str(self.clean_files))
        self.file_label.setWordWrap(True)
        self.sovits_lay.addWidget(self.file_label)
        self.file_button.clicked.connect(self.file_dialog)
        self.file_button.fileDropped.connect(self.update_files)

        self.input_preview = AudioPreviewWidget()
        self.sovits_lay.addWidget(self.input_preview)

        self.recent_label = QLabel("Recent Directories:")
        self.sovits_lay.addWidget(self.recent_label)
        self.recent_combo = QComboBox()
        self.sovits_lay.addWidget(self.recent_combo)
        self.recent_combo.activated.connect(self.recent_dir_dialog)

        self.transpose_validator = QIntValidator(-24,24)
        self.transpose_label = QLabel("Transpose")
        self.transpose_num = QLineEdit('0')
        self.transpose_num.setValidator(self.transpose_validator)

        self.ratio_validator = QDoubleValidator(0.0,1.0,2)
        self.ratio_label = QLabel("Retrieval Ratio")
        self.ratio_num = QLineEdit('0.0')
        self.ratio_num.setValidator(self.ratio_validator)
        self.ratio_frame = FieldWidget(
            self.ratio_label, self.ratio_num)
        self.sovits_lay.addWidget(self.ratio_frame)

        self.num_vec_validator = QIntValidator(1,10000)
        self.num_vec_label = QLabel("Number of retrieval vectors")
        self.num_vec_num = QLineEdit('3')
        self.num_vec_num.setValidator(self.ratio_validator)
        self.num_vec_frame = FieldWidget(
            self.num_vec_label, self.num_vec_num)
        self.sovits_lay.addWidget(self.num_vec_frame)

        self.f0_label = QLabel("f0 detection method")
        self.sovits_lay.addWidget(self.f0_label)
        self.f0_box = QComboBox()
        for f0_method in F0_OPTIONS:
            self.f0_box.addItem(f0_method)
        self.sovits_lay.addWidget(self.f0_box)

        self.transpose_frame = FieldWidget(
            self.transpose_label, self.transpose_num)
        self.sovits_lay.addWidget(self.transpose_frame)

        self.timestretch_validator = QDoubleValidator(0.5,1.0,3)

        if RUBBERBAND_AVAILABLE:
            self.ts_label = QLabel("Timestretch (0.5, 1.0)")
            self.ts_num = QLineEdit('1.0')
            self.ts_num.setValidator(self.timestretch_validator)

            self.ts_frame = FieldWidget(self.ts_label, self.ts_num)
            self.sovits_lay.addWidget(self.ts_frame)

        self.output_button = QPushButton("Change Output Directory")
        self.sovits_lay.addWidget(self.output_button)
        self.output_label = QLabel("Output directory: "+str(self.output_dir))
        self.sovits_lay.addWidget(self.output_label)
        self.output_button.clicked.connect(self.output_dialog)

        self.convert_button = QPushButton("Convert")
        self.sovits_lay.addWidget(self.convert_button)
        self.convert_button.clicked.connect(self.sofvits_convert)

        # TODO right now this only handles the first file processed.
        self.output_preview = AudioPreviewWidget()
        self.sovits_lay.addWidget(self.output_preview)

        self.sovits_lay.addStretch()

        self.delete_prep_cache = []

        self.audio_recorder_and_plugins = AudioRecorderAndVSTs(self)
        self.layout.addWidget(self.audio_recorder_and_plugins)

        # TalkNet component
        if self.talknet_available:
            self.try_load_talknet()

        self.update_recent_combo()

        if len(self.speakers):
            self.try_load_speaker(0)
        else:
            print("No speakers found!")

    def pass_editor_ctl(self, status : bool):
        self.setEnabled(not status)

    # Periodically delete junk files if mic_delfiles is toggled
    def try_delete_prep_cache(self):
        for f in self.delete_prep_cache:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    self.delete_prep_cache.remove(f)
                except PermissionError as e:
                    continue

    def update_files(self, files):
        if (files is None) or (len(files) == 0):
            return
        self.clean_files = files
        self.update_file_label()
        dir_path = os.path.abspath(os.path.dirname(self.clean_files[0]))
        if not dir_path in self.recent_dirs:
            self.recent_dirs.appendleft(dir_path)
        else:
            self.recent_dirs.remove(dir_path)
            self.recent_dirs.appendleft(dir_path)
        self.recent_combo.setCurrentIndex(self.recent_dirs.index(dir_path))
        self.update_input_preview()
        self.update_recent_combo()

    # Tests for TalkNet connection and compatibility
    def try_connect_talknet(self):
        import socket
        if not REQUESTS_AVAILABLE:
            print("requests library unavailable; not loading talknet options")
            return False
        spl = self.talknet_addr.split(':')
        if (spl is None) or (len(spl) == 1):
            print("Couldn't parse talknet address "+self.talknet_addr)
            return False
        ip = spl[0]
        port = int(spl[1])
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            result = sock.connect_ex((ip, port))
            if result == 0:
                print("TalkNet: Successfully found a service on address "
                      +self.talknet_addr)
                sock.close()
                return True
            else:
                print("Could not find TalkNet on address "+self.talknet_addr)
                sock.close()
                return False
        except socket.gaierror:
            print("Couldn't connect to talknet address "+self.talknet_addr)
            sock.close()
            return False
        except socket.error:
            print("Couldn't connect to talknet address "+self.talknet_addr)
            sock.close()
            return False
        sock.close()
        return False

    def try_load_talknet(self):
        self.talknet_frame = QGroupBox(self)
        self.talknet_frame.setTitle("talknet")
        self.talknet_frame.setStyleSheet("padding:10px")
        self.talknet_lay = QVBoxLayout(self.talknet_frame)

        self.character_box = QComboBox()
        self.character_label = QLabel("Speaker:")
        response = requests.get(
            'http://'+self.talknet_addr+'/characters',
            timeout=30)
        if response.status_code == 200:
            try:
                self.talknet_chars = json.loads(response.text)
            except Exception as e:
                self.talknet_available = False
                print("Couldn't parse TalkNet response.")
                print("Are you running the correct TalkNet server?")
                return

        for k in self.talknet_chars.keys():
            self.character_box.addItem(k)
        self.talknet_lay.addWidget(self.character_label)
        self.talknet_lay.addWidget(self.character_box)
        self.character_box.currentTextChanged.connect(self.talknet_character_load)
        if len(self.talknet_chars.keys()):
            self.cur_talknet_char = next(iter(self.talknet_chars.keys()))
        else:
            self.cur_talknet_char = "N/A"

        self.talknet_file_button = FileButton(label="Provide input audio")
        self.talknet_file = ""
        self.talknet_file_label = QLabel("File: "+self.talknet_file)
        self.talknet_file_label.setWordWrap(True)
        self.talknet_lay.addWidget(self.talknet_file_button)
        self.talknet_lay.addWidget(self.talknet_file_label)
        self.talknet_file_button.clicked.connect(self.talknet_file_dialog)
        self.talknet_file_button.fileDropped.connect(self.talknet_update_file)

        self.talknet_output_path = None

        self.talknet_input_preview = AudioPreviewWidget()
        self.talknet_lay.addWidget(self.talknet_input_preview)
       
        self.talknet_recent_label = QLabel("Recent Directories:")
        self.talknet_lay.addWidget(self.talknet_recent_label)
        self.talknet_recent_combo = QComboBox()
        self.talknet_lay.addWidget(self.talknet_recent_combo)
        self.talknet_recent_combo.activated.connect(self.talknet_recent_dir_dialog)

        self.talknet_transfer_sovits = FileButton(
            label='Transfer input to so-vits-svc')
        self.talknet_lay.addWidget(self.talknet_transfer_sovits)
        self.talknet_transfer_sovits.clicked.connect(self.transfer_to_sovits)

        self.talknet_transpose_label = QLabel("Transpose")
        self.talknet_transpose_num = QLineEdit('0')
        self.talknet_transpose_frame = FieldWidget(
            self.talknet_transpose_label, self.talknet_transpose_num)
        self.talknet_transpose_num.setValidator(self.transpose_validator)
        self.talknet_lay.addWidget(self.talknet_transpose_frame)

        self.talknet_transcript_label = QLabel("Transcript")
        self.talknet_transcript_edit = QPlainTextEdit()
        self.talknet_lay.addWidget(self.talknet_transcript_label)
        self.talknet_lay.addWidget(self.talknet_transcript_edit)

        self.talknet_dra = QCheckBox("Disable reference audio")
        self.talknet_lay.addWidget(self.talknet_dra)

        self.talknet_sovits = QCheckBox("Auto push TalkNet output to so-vits-svc")
        self.talknet_lay.addWidget(self.talknet_sovits)

        self.talknet_sovits_param = QCheckBox(
            "Apply left-side parameters to so-vits-svc gens")
        self.talknet_lay.addWidget(self.talknet_sovits_param)

        self.talknet_gen_button = QPushButton("Generate")
        self.talknet_lay.addWidget(self.talknet_gen_button)
        self.talknet_gen_button.clicked.connect(self.talknet_generate_request)

        self.talknet_output_info = QLabel("--output info (empty)--")
        self.talknet_output_info.setWordWrap(True)
        self.talknet_lay.addWidget(self.talknet_gen_button)
        self.talknet_lay.addWidget(self.talknet_output_info)

        self.talknet_manual = QPushButton(
            "Manual push TalkNet output to so-vits-svc section")
        self.talknet_lay.addWidget(self.talknet_manual)
        self.talknet_manual.clicked.connect(self.talknet_man_push_sovits)

        self.talknet_output_preview = AudioPreviewWidget()
        self.talknet_sovits_output_preview = AudioPreviewWidget()
        self.talknet_lay.addWidget(self.talknet_output_preview)
        self.talknet_lay.addWidget(self.talknet_sovits_output_preview)
        self.talknet_sovits_output_preview.hide()

        self.talknet_lay.setSpacing(0)
        self.talknet_lay.setContentsMargins(0,0,0,0)

        self.layout.addWidget(self.talknet_frame)
        print("Loaded TalkNet")

        # TODO ? multiple audio preview
        # TODO ? multiple audio selection for TalkNet?

        # TODO optional transcript output?
        # TODO option to disable automatically outputting sound files,
        # or to save in a separate directory.
        # TODO fancy concurrent processing stuff

    def talknet_character_load(self, k):
        self.cur_talknet_char = k

    def talknet_man_push_sovits(self):
        if self.talknet_output_path is None or not os.path.exists(self.talknet_output_path):
            return
        self.clean_files = [self.talknet_output_path]
        self.update_file_label()
        self.update_input_preview()

    def talknet_generate_request(self):
        req_time = datetime.now().strftime("%H:%M:%S")
        response = requests.post('http://'+self.talknet_addr+'/upload',
            data=json.dumps({'char':self.cur_talknet_char,
                'wav':self.talknet_file,
                'transpose':int(self.talknet_transpose_num.text()),
                'transcript':self.talknet_transcript_edit.toPlainText(),
                'results_dir':os.path.abspath(self.output_dir),
                'disable_reference_audio':self.talknet_dra.isChecked()}),
             headers={'Content-Type':'application/json'}, timeout=60)
        if response.status_code != 200:
            print("TalkNet generate request failed.")
            print("It may be useful to check the TalkNet server output.")
            return
        res = json.loads(response.text)

        if self.talknet_sovits.isChecked():
            if self.talknet_sovits_param.isChecked():
                sovits_res_path = self.convert([res["output_path"]])[0]
            else:
                sovits_res_path = self.convert([res["output_path"]],
                    dry_trans=0)[0]
        self.talknet_output_preview.from_file(res.get("output_path"))
        self.talknet_output_preview.set_text("Preview - "+res.get(
            "output_path","N/A"))
        self.talknet_output_path = res.get("output_path")
        if self.talknet_sovits.isChecked():
            self.talknet_output_preview.from_file(sovits_res_path)
            self.talknet_output_preview.set_text("Preview - "+
                sovits_res_path)
        self.talknet_output_info.setText("Last successful request: "+req_time+'\n'+
            "ARPAbet: "+res.get("arpabet","N/A")+'\n'+
            "Output path: "+res.get("output_path","N/A")+'\n')

    def update_file_label(self):
        self.file_label.setText("Files: "+str(self.clean_files))

    def update_input_preview(self):
        if not (PYGAME_AVAILABLE and self.mic_delfiles):
            self.input_preview.from_file(self.clean_files[0])
            self.input_preview.set_text("Preview - "+self.clean_files[0])

    def transfer_to_sovits(self):
        if (self.talknet_file is None) or not (
            os.path.exists(self.talknet_file)):
            return
        self.clean_files = [self.talknet_file]
        self.update_file_label()

    def try_load_speaker(self, index):
        print ("Loading "+self.speakers[index]["name"])
        self.speaker_label.setText("Speaker: "+self.speakers[index]["name"])
        print(self.speakers[index]["spk_path"])
        self.spk_emb = self.infer_tool.load_speaker_emb(
            self.speakers[index]["spk_path"])
        self.infer_tool.load_cluster(
            self.speakers[index]["cluster_path"])

        if self.speakers[index]["index_paths"]["hubert"] is None:
            self.ratio_frame.setEnabled(False)
            self.num_vec_frame.setEnabled(False)
        else:
            self.ratio_frame.setEnabled(True)
            self.num_vec_frame.setEnabled(True)

        if (self.speaker.get("model_path") is None or
            self.speakers[index]["model_path"] !=
            self.speaker["model_path"]):
            self.infer_tool.load_svc_model(
                self.speakers[index]["model_path"])

        self.speaker = self.speakers[index]

        pass

    def transfer_model_state(self, source, target):
        if source is None or target is None:
            return

    def talknet_file_dialog(self):
        self.talknet_update_file(
            QFileDialog.getOpenFileName(self, "File to process"))

    def talknet_update_preview(self):
        self.talknet_input_preview.from_file(self.talknet_file)
        self.talknet_input_preview.set_text("Preview - "+self.talknet_file)

    def talknet_update_file(self, files):
        if (files is None) or (len(files) == 0):
            return
        self.talknet_file = files[0]
        self.talknet_update_preview()
        self.talknet_file_label.setText("File: "+str(self.talknet_file))
        dir_path = os.path.abspath(os.path.dirname(self.talknet_file))
        if not dir_path in self.recent_dirs:
            self.recent_dirs.appendleft(dir_path)
        else:
            self.recent_dirs.remove(dir_path)
            self.recent_dirs.appendleft(dir_path)
        self.recent_combo.setCurrentIndex(self.recent_dirs.index(dir_path))
        self.update_recent_combo()

    def file_dialog(self):
        # print("opening file dialog")
        if not len(self.recent_dirs):
            self.update_files(QFileDialog.getOpenFileNames(
                self, "Files to process")[0])
        else:
            self.update_files(QFileDialog.getOpenFileNames(
                self, "Files to process", self.recent_dirs[0])[0])

    def recent_dir_dialog(self, index):
        # print("opening dir dialog")
        if not os.path.exists(self.recent_dirs[index]):
            print("Path did not exist: ", self.recent_dirs[index])
        self.update_files(QFileDialog.getOpenFileNames(
            self, "Files to process", self.recent_dirs[index])[0])

    def talknet_recent_dir_dialog(self, index):
        if not os.path.exists(self.recent_dirs[index]):
            print("Path did not exist: ", self.recent_dirs[index])
        self.talknet_update_file(QFileDialog.getOpenFileNames(
            self, "Files to process", self.recent_dirs[index])[0])

    def update_recent_combo(self):
        self.recent_combo.clear()
        if self.talknet_available:
            self.talknet_recent_combo.clear()
        for d in self.recent_dirs:
            self.recent_combo.addItem(backtruncate_path(d))
            if self.talknet_available:
                self.talknet_recent_combo.addItem(backtruncate_path(d))

    def output_dialog(self):
        temp_output_dir = QFileDialog.getExistingDirectory(self,
            "Output Directory", self.output_dir, QFileDialog.ShowDirsOnly)
        if not os.path.exists(temp_output_dir):
            return
        self.output_dir = temp_output_dir
        self.output_label.setText("Output Directory: "+str(self.output_dir))

        # int(self.transpose_num.text())

    def save_persist(self):
        with open(JSON_NAME, "w") as f:
            o = {"recent_dirs": list(self.recent_dirs),
                 "output_dir": self.output_dir}
            json.dump(o,f)

    def load_persist(self):
        if not os.path.exists(JSON_NAME):
            self.recent_dirs = []
            self.output_dir = "./results/"
            self.talknet_addr = TALKNET_ADDR
            return
        with open(JSON_NAME, "r") as f:
            o = json.load(f)
            self.recent_dirs = deque(o.get("recent_dirs",[]), maxlen=RECENT_DIR_MAXLEN)
            self.output_dir = o.get("output_dir",os.path.abspath("./results/"))
            self.talknet_addr = o.get("talknet_addr",TALKNET_ADDR)

    def sofvits_convert(self):
        res_paths = self.convert(self.clean_files)
        if len(res_paths) > 0 and not (PYGAME_AVAILABLE and self.mic_delfiles):
            self.output_preview.from_file(res_paths[0])
            self.output_preview.set_text("Preview - "+res_paths[0])
        return res_paths

    def convert(self, clean_files = [],
        dry_trans = None):
        res_paths = []
        
        ratio = float(self.ratio_num.text())
        n_retrieval_vec = int(self.num_vec_num.text())
        self.infer_tool.load_retrieval(
            self.speaker["index_paths"]["hubert"],
            self.speaker["index_paths"]["whisper"],
            ratio, n_retrieval_vec)

        trans = dry_trans
        if trans is None:
            trans = int(self.transpose_num.text())
        try:
            for clean_name in clean_files:
                clean_name = str(clean_name)
                audio_data = load_audio(clean_name)
                audio_data_32k = load_audio(clean_name, sr=32000)
                wav_name = Path(clean_name).stem

                if PEDALBOARD_AVAILABLE:
                    audio_data = self.audio_recorder_and_plugins.input_chain(
                        audio_data, audio_sr)

                if RUBBERBAND_AVAILABLE and (float(self.ts_num.text()) != 1.0):
                    audio_data = pyrb.time_stretch(
                        audio_data, audio_sr, float(self.ts_num.text()))

                f0_method = F0_OPTIONS[self.f0_box.currentIndex()]
                out_audio = self.infer_tool.infer(
                    audio_data, self.spk_emb, trans,
                    f0_method = f0_method, # Oops
                    x2_audio_data = audio_data_32k)
                
                res_path = os.path.join(self.output_dir,
                    f'{wav_name}_{trans}_'
                    f'{self.speaker["name"]}.{wav_format}')

                # Could be made more efficient
                i = 1
                while os.path.exists(res_path):
                    res_path = os.path.join(self.output_dir,
                        f'{wav_name}_{trans}key_'
                        f'{self.speaker["name"]}{i}.{wav_format}')
                    i += 1

                write(res_path,
                    self.infer_tool.hp.data.sampling_rate, out_audio)

                res_paths.append(res_path)

            if PYGAME_AVAILABLE and self.mic_state:
                if mixer.music.get_busy():
                    mixer.music.queue(res_paths[0])
                else:
                    mixer.music.load(res_paths[0])
                    mixer.music.play()
            if self.mic_delfiles:
                # Not sure how else to handle this without expensive loop
                self.delete_prep_cache.append(clean_name)
                self.delete_prep_cache.append(wav_path)
                self.delete_prep_cache.append(res_path)
                self.try_delete_prep_cache()
        except Exception as e:
            traceback.print_exc()
        return res_paths

if __name__ == "__main__":
    app = QApplication(sys.argv)

    parser = argparse.ArgumentParser()
    #parser.add_argument("--custom_merge", action="store_true",
        #help="Experimental support for weighted merge of"
        #" speakers within a model file")
    args = parser.parse_args()

    w = InferenceGui2(args)
    w.show()
    app.exec()
    w.save_persist()

