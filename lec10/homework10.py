import numpy as np
import torch, torch.nn

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step,
        then keep only the low-frequency half (the non-aliased half).
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    L = int(0.004 * Fs)
S = int(0.002 * Fs)

mstft = np.abs(
    np.fft.fft(
        np.array([
            waveform[m+1:m+1+L] - waveform[m:m+L]
            for m in range(0, len(waveform)-L, S)
        ]),
        axis=1
    )
)

features = 20 * np.log10(
    np.maximum(0.001 * np.amax(mstft), mstft)
)[:, 0:int(L/2)]

framelength = int(0.025 * Fs)
frameskip = int(0.01 * Fs)

energy = np.sum(
    np.square(
        np.array([
            waveform[m:m+framelength]
            for m in range(0, len(waveform)-framelength, frameskip)
        ])
    ),
    axis=1
)

VAD = [1 if energy[m] > 0.1 * max(energy) else 0 for m in range(len(energy))]

startframes = [m for m in range(1, len(VAD)) if VAD[m-1] == 0 and VAD[m] == 1]
endframes = [m for m in range(1, len(VAD)) if VAD[m-1] == 1 and VAD[m] == 0]

labels = np.zeros(len(features))

for (num, label) in [(1,'a'), (2,'i'), (3,'u'), (4,'e'), (5,'o')]:
    labels[5*startframes[num-1] : 5*endframes[num-1] + 4] = num

return features, labels
    '''
    

def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step.
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration of training

    The model should be Sequential(LayerNorm, Linear), 
    input dimension = NFEATS = number of columns in "features",
    output dimension = 1 + max(labels)

    The lossvalues should be computed using a CrossEntropy loss.
    '''
    model = torch.nn.Sequential(
    torch.nn.LayerNorm(features.shape[1]),
    torch.nn.Linear(features.shape[1], int(np.max(labels)) + 1)
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

lossvalues = []

for i in range(iterations):
    optimizer.zero_grad()
    outputs = model(features)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    lossvalues.append(loss.item())

return model, np.array(lossvalues)

def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array

    @return:
    probabilities (NFRAMES, NLABELS) - model output, transformed by softmax, detach().numpy().
    '''
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    return probabilities.detach().numpy()

