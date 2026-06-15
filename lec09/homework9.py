import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    frame_length = int(0.025 * Fs)
step = int(0.01 * Fs)

frames = np.array([
    waveform[m:m+frame_length]
    for m in range(0, len(waveform)-frame_length, step)
])

energy = np.sum(np.square(frames), axis=1)
vad = np.array([1 if e > 0.1*np.max(energy) else 0 for e in energy])

start = [m*step for m in range(1,len(vad)) if vad[m-1]==0 and vad[m]==1]
end = [m*step for m in range(1,len(vad)) if vad[m-1]==1 and vad[m]==0]

segments = [waveform[start[k]:end[k]] for k in range(len(start))]

return segments

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    models = []

N = int(0.004 * Fs)
S = int(0.002 * Fs)

for x in segments:
    frames = np.array([
        x[m+1:m+1+N] - x[m:m+N]
        for m in range(0, len(x)-N, S)
    ])

    mstft = np.abs(np.fft.fft(frames, axis=1))
    sgram = 20*np.log10(np.maximum(0.01*np.amax(mstft), mstft))

    model = np.average(sgram[:,0:int(N/2)], axis=0)
    models.append(model)

return models

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    testsegments = VAD(testspeech, Fs)
testspectra = segments_to_models(testsegments, Fs)

sims = np.zeros((len(models), len(testspectra)))

for y in range(len(models)):
    for k in range(len(testspectra)):
        x = testspectra[k]
        m = models[y]

        sims[y,k] = np.dot(x,m) / (
            np.sqrt(np.sum(np.square(x))) *
            np.sqrt(np.sum(np.square(m)))
        )

test_outputs = [
    labels[np.argmax(sims[:,k])]
    for k in range(len(testspectra))
]

return sims, test_outputs


