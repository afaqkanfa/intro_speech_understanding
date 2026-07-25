import gtts, speech_recognition, librosa, soundfile

def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech, then write it to filename.
    
    @params:
    text (str) - the text you want to synthesize
    lang (str) - the language in which you want to synthesize it
    filename (str) - the filename in which it should be saved
    '''
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)


def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files, and check their content using SpeechRecognition.
    The output files should be created as MP3, then converted to WAV, then recognized.

    @param:
    texts - a list of the texts you want to synthesize
    languages - a list of their languages
    filenames - a list of their root filenames, without the ".mp3" ending

    @return:
    recognized_texts - list of the strings that were recognized from each file
    '''
    recognized_texts = []

    for text, language, filename in zip(texts, languages, filenames):
        mp3_filename = filename + ".mp3"
        wav_filename = filename + ".wav"

        synthesize(text, language, mp3_filename)

        data, samplerate = librosa.load(mp3_filename)
        soundfile.write(wav_filename, data, samplerate)

        r = speech_recognition.Recognizer()

        with speech_recognition.AudioFile(wav_filename) as source:
            audio = r.record(source)
            recognized_text = r.recognize_google(audio, language=language)

        recognized_texts.append(recognized_text)

    return recognized_texts
