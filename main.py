from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, QThread
from PyQt5.QtWidgets import QApplication, QFileDialog
import librosa
import matplotlib.pyplot as plt
import sys
import ui
import soundfile as sf
# from playsound import playsound
# import multiprocessing
import winsound

class MainFrame(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainFrame, self).__init__(parent)
        self.setupUi(self)

        self.audio_edited = None
        self.audio = None
        self.sr = None
        self.load_audio.clicked.connect(self.open_file_dialog_and_read_audio)
        self.play.clicked.connect(self.play_audio)
        self.pause.clicked.connect(self.pause_audio)
        self.pitch_spin.valueChanged.connect(self.pitch_shift)
        self.speed_spin.valueChanged.connect(self.speed_shift)
        

    def open_file_dialog_and_read_audio(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Choose your file", r"C:\Users\fumchin\Music","All Files (*);;MP3 Files (*.mp3);;WAV Files (*.wav)", options=options)
        self.audio, self.sr = librosa.load(fileName)
        self.audio_edited = self.audio

    def play_audio(self):
        sf.write('temp.wav', self.audio_edited, self.sr, 'PCM_24')
        winsound.PlaySound('temp.wav', winsound.SND_ASYNC)
        
    def pause_audio(self):
        winsound.PlaySound(None, winsound.SND_PURGE)

    def pitch_shift(self):
        self.audio_edited = librosa.effects.pitch_shift(self.audio, self.sr, n_steps=self.pitch_spin.value())

    def speed_shift(self):
        self.audio_edited = librosa.effects.time_stretch(self.audio, rate=self.speed_spin.value())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = MainFrame()
    form.show()
    app.exec_()