import os
import logging
import numpy as np
import copy

from scipy.interpolate import interp1d
from scipy.signal import firwin
from scipy.signal import lfilter
from multiprocessing import Pool
from multiprocessing import cpu_count
#from sprocket.speech import FeatureExtractor


def load_from_file(path, dimension):
    data = np.fromfile(path, dtype=np.float32)
    if len(data) % dimension != 0:
        raise RuntimeError('%s data size is not divided by %d'%(path, dimension))
    data = data.reshape([-1, dimension])
    return data


def save_to_file(data, path):
    data.astype(np.float32).tofile(path)


def _lf02vuv(data):
    '''
    generate vuv feature by interpolating lf0
    '''
    data = np.reshape(data, (data.size, 1))

    vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
    vuv_vector[data > 0.0] = 1.0
    vuv_vector[data <= 0.0] = 0.0

    ip_data = data

    frame_number = data.size
    last_value = 0.0
    for i in range(frame_number):
        if data[i] <= 0.0:
            j = i+1
            for j in range(i+1, frame_number):
                if data[j] > 0.0:
                    break
            if j < frame_number-1:
                if last_value > 0.0:
                    step = (data[j] - data[i-1]) / float(j - i + 1)
                    for k in range(i, j):
                        ip_data[k] = data[i-1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        ip_data[k] = data[j]
            else:
                for k in range(i, frame_number):
                    ip_data[k] = last_value
        else:
            ip_data[i] = data[i]
            last_value = data[i]

    return ip_data, vuv_vector


def _conv1d(data_matrix, kernel):
    '''
    convolve each column in data_matrix with kernel
    类似CNN的那种1d卷积
    '''
    kernel = kernel.reshape([-1, ])
    kernel_width = int(len(kernel) / 2)

    res = []
    for dim in range(data_matrix.shape[1]):
        vector = data_matrix[:, dim].reshape([-1, ])
        vector = np.pad(vector, (kernel_width, kernel_width), 'edge')
        res.append(np.correlate(vector, kernel, mode='valid').reshape([-1,1]))

    res = np.concatenate(res, axis=-1)
    return res



def extract_feats(world_analysis, wav_dir, feat_dir, filename, mgc_dim=60):
    world_analysis_cmd = "{analyze} {wav} {lf0} {mgc} {bap} {mgc_dim}".format(analyze=world_analysis,
                                                                              wav=os.path.join(wav_dir, filename + '.wav'),
                                                                              lf0=os.path.join(feat_dir, filename + '.lf0'),
                                                                              mgc=os.path.join(feat_dir, filename + '.mgc'),
                                                                              bap=os.path.join(feat_dir, filename + '.bap'),
                                                                              mgc_dim=mgc_dim)


def _merge_feat(feat_dir, out_dir, filenames):
    '''
    merge acoustic features
    最终生成的特征为[lf0, lf0与delta的卷积， lf0与acc的卷积， mgc， mgc与delta的卷积， mgc与acc的卷积， bap， bap与delta的卷积，
    bap与acc的卷积， vuv]
    '''
    for filename in filenames:
        lf0_path = os.path.join(feat_dir, filename + '.lf0')
        mgc_path = os.path.join(feat_dir, filename + '.mgc')
        bap_path = os.path.join(feat_dir, filename + '.bap')
        out_path = os.path.join(out_dir, filename + '.cmp')

        lf0_matrix = load_from_file(lf0_path, 1)
        mgc_matrix = load_from_file(mgc_path, 1)
        bap_matrix = load_from_file(bap_path, 1)

        frame_num = lf0_matrix.shape[0]
        mgc_matrix = mgc_matrix.reshape([frame_num, -1])
        bap_matrix = bap_matrix.reshape([frame_num, -1])

        lf0_matrix, vuv_matrix = _lf02vuv(lf0_matrix)

        delta_win = np.array([-0.5, 0.0, 0.5])
        acc_win = np.array([1.0, -2.0, 1.0])
        res = []
        res.append(lf0_matrix)
        res.append(_conv1d(lf0_matrix, delta_win))
        res.append(_conv1d(lf0_matrix, acc_win))
        res.append(mgc_matrix)
        res.append(_conv1d(mgc_matrix, delta_win))
        res.append(_conv1d(mgc_matrix, acc_win))
        res.append(bap_matrix)
        res.append(_conv1d(bap_matrix, delta_win))
        res.append(_conv1d(bap_matrix, acc_win))
        res.append(vuv_matrix)
        res = np.concatenate(res, axis=-1)

        save_to_file(res, out_path)

    return lf0_matrix.shape[1] * 3, mgc_matrix.shape[1] * 3, bap_matrix.shape[1] * 3, vuv_matrix.shape[1]


def wav_preprocess(data_dir, tmp_dir, world_dir):
    '''
    从音频中提取特征，并将他们合起来，计算新特征
    '''
    logger = logging.getLogger('preprocess')
    logger.setLevel(logging.INFO)

    wav_dir = os.path.join(data_dir, 'wavs')
    feat_dir = os.path.join(tmp_dir, 'feats')
    cmp_dir = os.path.join(tmp_dir, 'cmp')
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(cmp_dir, exist_ok=True)

    filenames = list(set(filename.split('.')[0] for filename in os.listdir(wav_dir)))
    split_filenames = [filenames[i::cpu_count()] for i in range(cpu_count())]
    world_analysis = os.path.join(world_dir, 'analysis')

    # 使用world提取特征
    logger.info('extract feat from wav')
    p = Pool(cpu_count())
    results = []
    for filename in filenames:
        results.append(p.apply_async(extract_feats, args=[world_analysis, wav_dir, feat_dir, filename]))
    p.close()
    p.join()
    results = [res.get() for res in results]

    # 将lf0，mgc，bap合起来，并得到新特征
    logger.info('merge lf0 mgc bap feat')
    p = Pool(cpu_count())
    results = []
    for filenames in split_filenames:
        results.append(p.apply_async(_merge_feat, args=[feat_dir, cmp_dir, filenames]))
    p.close()
    p.join()

    logger.info('preprocess wav finish')


def world_feature_extract(wav, config):
    """WORLD feature extraction

    Args:
        queue (multiprocessing.Queue): the queue to store the file name of utterance
        wav_list (list): list of the wav files
        config (dict): feature extraction config

    """
    # define feature extractor
    feature_extractor = FeatureExtractor(
        analyzer="world",
        fs=config['sampling_rate'],
        shiftms=config['hop_size'] / config['sampling_rate'] * 1000,
        minf0=config['minf0'],
        maxf0=config['maxf0'],
        fftl=config['fft_size'])
    # extraction

        # extract features
    f0, spc, ap = feature_extractor.analyze(wav)
    codeap = feature_extractor.codeap()
    mcep = feature_extractor.mcep(dim=config['mcep_dim'], alpha=config['mcep_alpha'])
    npow = feature_extractor.npow()
    uv, cont_f0 = convert_continuos_f0(f0)
    lpf_fs = int(config['sampling_rate'] / config['hop_size'])
    cont_f0_lpf = low_pass_filter(cont_f0, lpf_fs, cutoff=20)
    next_cutoff = 70
    while not (cont_f0_lpf >= [0]).all():
        cont_f0_lpf = low_pass_filter(cont_f0, lpf_fs, cutoff=next_cutoff)
        next_cutoff *= 2
    # concatenate
    cont_f0_lpf = np.expand_dims(cont_f0_lpf, axis=-1)
    uv = np.expand_dims(uv, axis=-1)
    feats = np.concatenate([uv, cont_f0_lpf, mcep, codeap], axis=1)

    # return (feats, f0, ap, spc, npow)
    return feats

def convert_continuos_f0(f0):
    """Convert F0 to continuous F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)
    Return:
        (ndarray): continuous f0 with the shape (T)

    """
    # get uv information as binary
    uv = np.float32(f0 != 0)
    # get start and end of f0
    if (f0 == 0).all():
        logging.warn("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    # padding start and end of f0 sequence
    cont_f0 = copy.deepcopy(f0)
    start_idx = np.where(cont_f0 == start_f0)[0][0]
    end_idx = np.where(cont_f0 == end_f0)[0][-1]
    cont_f0[:start_idx] = start_f0
    cont_f0[end_idx:] = end_f0
    # get non-zero frame index
    nz_frames = np.where(cont_f0 != 0)[0]
    # perform linear interpolation
    f = interp1d(nz_frames, cont_f0[nz_frames])
    cont_f0 = f(np.arange(0, cont_f0.shape[0]))

    return uv, cont_f0


def low_pass_filter(x, fs, cutoff=70, padding=True):
    """Low pass filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter
    Return:
        (ndarray): Low pass filtered waveform sequence

    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

    return lpf_x
