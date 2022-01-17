import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import soundfile as sf
import math


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, hparams):
    wav = wav / np.abs(wav).max() * 0.999
    f1 = 0.5 * 32767 / max(0.01, np.max(np.abs(wav)))
    f2 = np.sign(wav) * np.power(np.abs(wav), 0.95)
    wav = f1 * f2
    wav = signal.convolve(wav, signal.firwin(hparams['num_freq'],
                                             [hparams['fmin'], hparams['fmax']],
                                             pass_zero=False,
                                             fs=hparams['audio_sample_rate']))
    # proposed by @dsmiller
    wavfile.write(path, hparams['audio_sample_rate'], wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def save_melGAN_wav(file_path, sampling_rate, audio):
    audio = audio.reshape((-1, ))
    sf.write(file_path,
             audio, sampling_rate, "PCM_16")


def preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)


def inv_preemphasis(wav, k):
    return signal.lfilter([1], [1, -k], wav)


def trim_silence(wav, hparams, only_front=True):
    non_silent = librosa.effects._signal_to_frame_nonsilent(wav,
                                            frame_length=hparams['trim_fft_size'],
                                            hop_length=hparams['trim_hop_size'],
                                            ref=np.max,
                                            top_db=hparams['trim_top_db'])

    nonzero = np.flatnonzero(non_silent)
    if nonzero.size > 0:
        # Compute the start and end positions
        # End position goes one frame past the last non-zero
        start = int(librosa.core.frames_to_samples(nonzero[0], hparams['trim_hop_size']))
        end = min(wav.shape[-1],
                  int(librosa.core.frames_to_samples(nonzero[-1] + 1, hparams['trim_hop_size'])))
    else:
        # The signal only contains zeros
        start, end = 0, 0
    if only_front:
        end = wav.shape[0]
    full_index = [slice(None)] * wav.ndim
    full_index[-1] = slice(start, end)

    return wav[tuple(full_index)]


def get_hop_size(hparams):
    hop_size = hparams['hop_size']
    if hop_size is None:
        assert hparams['frame_shift_ms'] is not None
        hop_size = int(hparams['frame_shift_ms'] / 1000 * hparams['sampling_rate'])
    return hop_size


def inv_linear_spectrogram(linear_spectrogram, hparams):
    '''Converts linear spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram

    S = _db_to_amp(D + hparams.ref_level_db)  # Convert back to linear

    return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis)


def _griffin_lim(S, hparams):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y


def _stft(y, hparams):
    return librosa.stft(y=y, n_fft=hparams['fft_size'], hop_length=get_hop_size(hparams),
                        win_length=hparams['win_length'])

def _istft(y, hparams):
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams['win_length'])

# Conversions
_mel_basis = None
_inv_mel_basis = None


def _build_mel_basis(hparams):
    assert hparams['fmax'] <= hparams['sampling_rate'] // 2
    return librosa.filters.mel(hparams['sampling_rate'], hparams['fft_size'], n_mels=hparams['num_mels'],
                               fmin=hparams['fmin'], fmax=hparams['fmax'])#,norm=None if hparams['use_same_high'] else 1)


def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _amp_to_db(x, hparams):
    min_level = np.exp(hparams['min_level_db'] / 20 * np.log(10))  # np.log()以e为底，np.exp()返回e的幂次方
    return 20 * np.log10(np.maximum(min_level, x))  # np.maximum逐位返回两个参数较大值


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S, hparams):
    if hparams['allow_clipping_in_normalization']:
        if hparams['symmetric_mels']:
            return np.clip((2 * hparams['max_abs_value']) * (
                        (S - hparams['min_level_db']) / (-hparams['min_level_db'])) - hparams['max_abs_value'],
                           -hparams['max_abs_value'], hparams['max_abs_value'])
        else:
            return np.clip(hparams['max_abs_value'] * ((S - hparams['min_level_db']) / (-hparams['min_level_db'])), 0,
                           hparams['max_abs_value'])

    if hparams['symmetric_mels']:
        return (2 * hparams['max_abs_value']) * (
                    (S - hparams['min_level_db']) / (-hparams['min_level_db'])) - hparams['max_abs_value']
    else:
        return hparams['max_abs_value'] * ((S - hparams['min_level_db']) / (-hparams['min_level_db']))


def _denormalize(D, hparams):
    if hparams['allow_clipping_in_normalization']:
        if hparams['symmetric_mels']:
            return (((np.clip(D, -hparams['max_abs_value'],
                              hparams['max_abs_value']) + hparams['max_abs_value']) * -hparams['min_level_db'] / (
                                 2 * hparams['max_abs_value']))
                    + hparams['min_level_db'])
        else:
            return ((np.clip(D, 0,
                             hparams['max_abs_value']) * -hparams['min_level_db'] / hparams['max_abs_value']) + hparams['min_level_db'])

    if hparams['symmetric_mels']:
        return (((D + hparams['max_abs_value']) * -hparams['min_level_db'] / (
                    2 * hparams['max_abs_value'])) + hparams['min_level_db'])
    else:
        return ((D * -hparams['min_level_db'] / hparams['max_abs_value']) + hparams['min_level_db'])


def linearspectrogram(wav, hparams):
    if hparams['preemphasis']:
        wav = preemphasis(wav, hparams['preemphasis_value'])
    D = _stft(wav, hparams)
    S = _amp_to_db(np.abs(D), hparams) - hparams['ref_level_db']

    if hparams['signal_normalization']:
        return _normalize(S, hparams)
    return S


def melspectrogram(wav, hparams):
    if hparams['preemphasis']:
        wav = preemphasis(wav, hparams['preemphasis_value'])
    D = _stft(wav, hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams['ref_level_db']

    if hparams['signal_normalization']:
        return _normalize(S, hparams)
    return S


def logmelfilterbank(audio, config, eps=1e-10):

    x_stft = librosa.stft(audio, n_fft=config["fft_size"], hop_length=config["hop_size"],  # stft变换
                          win_length=config["win_length"], window=config["window"], pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis  得到mel偏移量
    mel_basis = librosa.filters.mel(sr=config["sampling_rate"], n_fft=config["fft_size"],
                                    n_mels=config["num_mels"], fmin=config["fmin"], fmax=config["fmax"])
                                    # norm=None if config['use_same_high_mel'] else 1)

    return 20 * np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))


# def chroma_stft(x, hparams):
#
#     S = np.abs(_stft(x, hparams))**2
#
#     tuning = estimate_tuning(S=S, sr=hparams['sampling_rate'], bins_per_octave=12)
#
#     # Get the filter bank
#     chromafb = filters.chroma(hparams['sampling_rate'], hparams['fft_size'],
#                               tuning=tuning, n_chroma=12)
#
#     # Compute raw chroma
#     raw_chroma = np.dot(chromafb, S)
#
#     return raw_chroma


def inv_mel_spectrogram(mel_spectrogram, hparams):
    '''Converts mel spectrogram to waveform using librosa'''
    if hparams['signal_normalization']:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram

    S = _mel_to_linear(_db_to_amp(D + hparams['ref_level_db']), hparams)  # Convert back to linear

    return inv_preemphasis(_griffin_lim(S ** hparams['power'], hparams), hparams['preemphasis'])


# waveRNN wav funcation
def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)

def decode_mu_law(y, mu, from_labels=True) :
    # TODO : get rid of log2 - makes no sense
    if from_labels : y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x

def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2 ** bits - 1) / 2
    return x.clip(0, 2 ** bits - 1)

def label_2_float(x, bits):
    return 2 * x / (2 ** bits - 1.) - 1.


def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


import matplotlib.pyplot as plt


def plot_spec(spec, path, info=None):
    fig = plt.figure(figsize=(14, 7))
    heatmap = plt.pcolor(spec)
    fig.colorbar(heatmap)

    xlabel = 'Time'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Mel filterbank')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close(fig)

# Compute the mel scale spectrogram from the wav
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return np.exp(x) / C


def pitchfeats(wav,hparams):  # 提取pitch特征

    pitches,magnitudes =librosa.piptrack(wav,hparams['sampling_rate'],
                                         n_fft=hparams['fft_size'],
                                         hop_length=hparams['hop_size'],
                                         fmin=hparams['fmin'],
                                         fmax=2000,
                                         win_length=hparams['win_length'])
    pitches = pitches.T
    magnitudes = magnitudes.T
    assert pitches.shape==magnitudes.shape

    pitches = [pitches[i][find_f0(magnitudes[i])] for i,_ in enumerate(pitches) ]  # 寻找pitches二维向量中最大值

    return np.asarray(pitches)


def find_f0(mags):
    tmp=0
    mags=list(mags)
    for i,mag in enumerate(mags):
        if mag < tmp: # 若赋值<0:
            # return i-1
            if tmp-mag>2:  # 若赋值<2+tmp
                #return i-1
                return mags.index(max(mags[0:i]))  #返回最大值所在下下标
            else:
                return 0
        else:  # 若赋值>0:令tmp = mag
            tmp = mag
    return 0


def f0_to_coarse(f0, f0_min=35, f0_max=1400, f0_bin = 256):

    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    # f0_mel[f0_mel == 0] = 0
    # 大于0的分为255个箱
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel < 0] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = np.rint(f0_mel).astype(np.int)
    # print('Max f0', np.max(f0_coarse), ' ||Min f0', np.min(f0_coarse))
    assert (np.max(f0_coarse) <= 256 and np.min(f0_coarse) >= 0)
    return f0_coarse
