import tensorflow
from django.shortcuts import render
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import speech_recognition as sr
from playsound import playsound
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


def redirect_mainpage(request):
    return render(request, 'mainpage.html')


SAMPLES_TO_CONSIDER = 22050


class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """

    model = load_model("model.h5")
    # _mapping = ['wow', 'bird', 'cat', 'sheila', 'nine', 'dog',
    #             'go', 'eight', 'house', 'one', 'marvin', 'five',
    #             'yes', 'left', 'down', 'three', 'no', 'happy',
    #             'on', 'stop', 'tree', 'zero', 'seven', 'six',
    #             'bed', 'up', 'off', 'right', 'two', 'four']

    _mapping = [
        'down',
        "off",
        "on",
        "no",
        "yes",
        "stop",
        "up",
        "right",
        "left",
        "go"
    ]

    _instance = None


    def predict(self, file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tensorflow.keras.models.load_model('model.h5')
    return _Keyword_Spotting_Service._instance


test_name = ''


def test_pass(request):
    global test_name
    tensorflow.compat.v1.disable_eager_execution()
    if request.method == 'POST':
        try:
            test_audio = request.FILES["audiofile"]
        except:
            return render(request, 'mainpage.html', {'error_msg': 'Kindly select a wav file first'})

        # create 2 instances of the keyword spotting service
        kss = Keyword_Spotting_Service()
        kss1 = Keyword_Spotting_Service()
        # check that different instances of the keyword spotting service point back to the same object (singleton)
        assert kss is kss1

        # make a prediction
        keyword = kss.predict(test_audio)
        print(keyword)
        with open('audio/' + test_audio.name, 'wb+') as destination:
            for chunk in test_audio.chunks():
                destination.write(chunk)
        test_name = test_audio.name

        return render(request, 'mainpage.html', {'prediction': keyword})


r = sr.Recognizer()
mic_name = "USB Device 0x46d:0x825: Audio (hw:1, 0)"
sample_rate = 48000

chunk_size = 2048
mic_list = sr.Microphone.list_microphone_names()
device_id = 0
for i, microphone_name in enumerate(mic_list):
    if microphone_name == mic_name:
        device_id = i


def speech_google(request):
    with sr.Microphone(device_index=device_id, sample_rate=sample_rate,
                       chunk_size=chunk_size) as source:
        r.adjust_for_ambient_noise(source)
        print("Say Something")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return render(request, 'mainpage.html', {'predict_google': text})

        except sr.UnknownValueError:
            return render(request, 'mainpage.html',
                          {'predict_google': "Google Speech Recognition could not understand audio"})

        except sr.RequestError as e:
            return render(request, 'mainpage.html',
                          {'predict_google': "Could not request results from Google SpeechRecognitionservice;{0}".format(e)})


def play_audio(request):
    try:
        playsound('audio/' + test_name)
    except:
        return render(request, 'mainpage.html', {'play_msg': 'Kindly select a audio command from above'})
    return render(request, 'mainpage.html')
