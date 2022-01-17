
## Mel-filterbank
mel_n_channels = 80
win_length = 512
hop_length = 128
n_fft = 512
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds

## Audio
sampling_rate = 24000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 240     # 2400 ms
# Number of spectrogram frames at inference
inference_n_frames = 120     #  1200 ms


## Voice Activation Detection
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 20  # In milliseconds
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out. 
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6


## Audio volume normalization
audio_norm_target_dBFS = -30
