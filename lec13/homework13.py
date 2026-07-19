import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.

    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute

    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''
    nframes = int((len(speech) - frame_length) / frame_skip)

    frames = np.zeros((nframes, frame_length))
    for frame in range(nframes):
        frames[frame] = speech[frame * frame_skip:frame * frame_skip + frame_length]

    A = librosa.lpc(frames, order=order)

    excitation = np.zeros((nframes, frame_length))
    for frame in range(nframes):
        for samp in range(order, frame_length):
            for k in range(order + 1):
                excitation[frame, samp] += A[frame, k] * frames[frame, samp - k]

    return A, excitation


def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.

    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples

    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    nframes = len(A)
    order = A.shape[1] - 1

    synthesis = np.zeros(len(e))

    for n in range(len(e)):
        frame = int(n / frame_skip)
        synthesis[n] = e[n]
        for k in range(1, min(n, order + 1)):
            synthesis[n] -= A[frame, k] * synthesis[n - k]

    return synthesis


def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.

    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples

    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
    gain = np.sqrt(np.average(np.square(excitation), axis=1))

    e_robot = np.zeros(len(gain) * frame_skip)

    n = 0
    while n < len(e_robot):
        gain_frame = min(int(n / frame_skip), len(gain) - 1)
        e_robot[n] = gain[gain_frame]
        n += T0

    return gain, e_robot
