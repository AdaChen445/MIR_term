from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, QThread
from PyQt5.QtWidgets import QApplication, QFileDialog
import librosa
from src.infer import audio_infer
import matplotlib.pyplot as plt
import sys
import ui
import soundfile as sf
import winsound
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
import skimage
import cv2
import numpy as np
import os
import pickle

class MainFrame(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self, parent=None): #edited
        super(MainFrame, self).__init__(parent)
        self.setupUi(self)

        if not os.path.isdir('temp'): os.mkdir('temp')
        self.audio_edited = None
        self.audio = None
        self.sr = None
        self.model = load_model('best_0.8271.h5')
        self.le = pickle.loads(open('le.pickle', 'rb').read())

        self.load_audio.clicked.connect(self.open_file_dialog_and_read_audio)
        self.export_audio.clicked.connect(self.export_wav)
        self.play.clicked.connect(self.play_audio)
        self.pause.clicked.connect(self.pause_audio)
        self.pitch_spin.valueChanged.connect(self.pitch_shift)
        self.speed_spin.valueChanged.connect(self.speed_shift)
        self.gc.clicked.connect(self.genre_classify)

        # for timbre transfer==========================================================
        self.transfer_button.clicked.connect(self.transfer)
        self.timbre_path = r"./timbre_transfer_weight"
        self.recover_button.clicked.connect(self.timbre_recover)
        self.pre_transfer_audio = None

        

        

    def open_file_dialog_and_read_audio(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Choose your file", r"C:\Users\fumchin\Music","All Files (*);;MP3 Files (*.mp3);;WAV Files (*.wav)", options=options)
        self.audio, self.sr = librosa.load(fileName, sr=22050)
        self.audio_edited = self.audio

    def play_audio(self): #edited
        sf.write('temp/temp.wav', self.audio_edited, self.sr, 'PCM_24')
        winsound.PlaySound('temp/temp.wav', winsound.SND_ASYNC)
        
    def pause_audio(self):
        winsound.PlaySound(None, winsound.SND_PURGE)

    def pitch_shift(self): #edited
        self.audio_edited = librosa.effects.pitch_shift(self.audio, self.sr, n_steps=self.pitch_spin.value())
        self.audio_edited = librosa.effects.time_stretch(self.audio_edited, rate=self.speed_spin.value())

    def speed_shift(self): #edited
        self.audio_edited = librosa.effects.time_stretch(self.audio, rate=self.speed_spin.value())
        self.audio_edited = librosa.effects.pitch_shift(self.audio_edited, self.sr, n_steps=self.pitch_spin.value())

    def export_wav(self): #new added
        name = self.export_name.text()+'.wav'
        print(name)
        sf.write(name, self.audio_edited, self.sr, 'PCM_24')

    def genre_classify(self): #new added
        img = self.stack3channels()
        pred = self.model.predict(img)
        normalized_pred = pred*1/np.sum(pred)
        pred_ind = np.argsort(-pred, axis=1)
        self.gc_result.setText(str(self.le.classes_[pred_ind[0][0]])+' '+str(normalized_pred[0][pred_ind[0][0]])+'\n'+
                str(self.le.classes_[pred_ind[0][1]])+' '+str(normalized_pred[0][pred_ind[0][1]])+'\n'+
                str(self.le.classes_[pred_ind[0][2]])+' '+str(normalized_pred[0][pred_ind[0][2]]))  

    def audio2feature(self, y, sr, nfft): #new added
        y=y[sr*5:sr*15]
        D1 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=nfft, win_length=512)
        D1 = np.log(D1 + 1e-9)
        D1 = scale_minmax(D1, 0, 255).astype(np.uint8)
        D1 = np.flip(D1, axis=0)
        D1 = 255-D1
        D2 = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=nfft, win_length=512)
        D2 = scale_minmax(D2, 0, 255).astype(np.uint8)
        D2 = 255-D2
        img = np.concatenate((D1,D2), axis=0)
        name = 'temp/temp_'+str(nfft)+'.png'
        skimage.io.imsave(name, img)
        img = cv2.imread(name,  cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (431, 140))
        img = np.array(img, dtype=np.uint8)
        return img

    def stack3channels(self): #new added
        grc_audio = self.audio_edited[10*self.sr:20*self.sr]
        n1024 = self.audio2feature(grc_audio, self.sr, 1024)
        n2048 = self.audio2feature(grc_audio, self.sr, 2048)
        n4096 = self.audio2feature(grc_audio, self.sr, 4096)
        img = np.dstack([n1024, n2048, n4096]).astype(np.uint8)
        img = np.array(img, dtype=np.float32)/255.0
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def transfer(self):
        timbre_dict = {"Trumpet":1, "Violin":2, "Acoustic Guitar":3}
        # source_label = self.source_comboBox.currentText()
        self.pre_transfer_audio = self.audio_edited
        sf.write('temp/temp.wav', self.audio_edited, self.sr, 'PCM_24')
        target_label = self.target_comboBox.currentText()
        # source_index = timbre_dict[source_label]
        target_index = timbre_dict[target_label]
        audio_infer('temp/temp.wav', target_index)
        self.audio_edited, self.sr = librosa.load('temp/temp.wav')
        # print(source_label)

    def timbre_recover(self):
        self.audio_edited = self.audio
        sf.write('temp/temp.wav', self.audio_edited, self.sr, 'PCM_24')


def scale_minmax(X, min=0.0, max=1.0): #new added
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled



if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = MainFrame()
    form.show()
    app.exec_()