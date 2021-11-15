import numpy as np
from scipy.fft import fft
from scipy.signal import stft, medfilt
import pandas as p
import framework.extension.math as me


def find_fdr(x, fs: int,
             frame_length: int, frame_overlap: int,
             sub_frame_length: int, sub_frame_overlap: int,
             bands, band_multi, nfft: int, L_lim: int
             ) -> []:
    sub_frames_num = int(len(x) / (sub_frame_length - sub_frame_overlap)) - \
                     int(sub_frame_length / (sub_frame_length - sub_frame_overlap))
    frames_num = int(sub_frames_num / (frame_length - frame_overlap)) or 1

    x_fft_list = []
    ham_win = np.hamming(sub_frame_length)
    for i in range(sub_frames_num):
        # windowing
        if i == sub_frames_num - 1:
            x_frame = x[int(i * (sub_frame_length - sub_frame_overlap)): -1]
            ham_win = np.hamming(len(x_frame))
            x_win = x_frame * ham_win
        else:
            x_frame = x[int(i * (sub_frame_length - sub_frame_overlap)):
                        int(i * (sub_frame_length - sub_frame_overlap)) + sub_frame_length]
            x_win = x_frame * ham_win
        # going in Fourier domain
        x_fft = fft(x_win, n=nfft)
        x_fft = abs(x_fft[0: int(nfft / 2) - 1])
        x_fft_list.append(abs(x_fft))

    ret = []
    for fr in range(frames_num):
        if fr == frames_num - 1:
            frame_x_fft_list = x_fft_list[int(fr * (frame_length - frame_overlap)): -1]
        else:
            frame_x_fft_list = x_fft_list[int(fr * (frame_length - frame_overlap)):
                                          int(fr * (frame_length - frame_overlap)) + frame_length]

        signal_energy = sum([sum(pow(frame_x_fft, 2)) for frame_x_fft in frame_x_fft_list])

        frames_to_be_plotted = int(frame_length / 5)
        if fr == frames_num - 1:
            frame_x_signal = x[int(fr * (frame_length - frame_overlap) * (sub_frame_length - sub_frame_overlap)): -1]
        else:
            frame_x_signal = x[int(fr * (frame_length - frame_overlap) * (sub_frame_length - sub_frame_overlap)):
                               int(fr * (frame_length - frame_overlap) * (sub_frame_length - sub_frame_overlap)) +
                                         frames_to_be_plotted * (sub_frame_length - sub_frame_overlap) + sub_frame_overlap]
        x_max = max(frame_x_signal)
        frame_x_signal = [j / x_max for j in frame_x_signal]

        bands_fdrs = []
        # COMPUTING SUB-BAND DECAY RATES
        for band in bands:
            band_fdrs = []

            # [third-]octave band case
            lower_bin = int(np.floor((band / band_multi * nfft) / fs))
            upper_bin = int(np.ceil((band * band_multi * nfft) / fs))

            # finding the energy envelope on the current frequency band
            env = []
            for x_fft in frame_x_fft_list:
                env_val = 0
                for j in range(lower_bin, upper_bin):
                    env_val = env_val + pow(x_fft[j], 2)
                env.append(env_val)

            # total energy in current band
            band_energy = sum(env)
            # en_octave_band.append(band_energy)
            if band_energy > signal_energy * 0.001:
                down_threshold = 1E-3 * max(env)

                # Searching for subsequent frames with decreasing energy
                found = False
                search_can_end = False
                dec_frames = []
                dec_frames_num = L_lim
                dec_frames_num_min = 3
                while not search_can_end:
                    i = 0
                    while i < len(env):
                        frames_counter = 0
                        dec_frames.append(env[i])
                        k = 0
                        for k in range(i, len(env) - 1):
                            if (env[k] - env[k + 1]) >= down_threshold:
                                dec_frames.append(env[k + 1])
                                frames_counter += 1
                            elif (env[k] - env[k + 1] < down_threshold) or (env[k + 1] == 0) or \
                                    (k + 1 == (len(env) - 1)):
                                if frames_counter >= dec_frames_num:
                                    bat = i * (sub_frame_length - sub_frame_overlap)
                                    eat = k * (sub_frame_length - sub_frame_overlap) + sub_frame_length
                                    band_fdrs.append({
                                        "begin_at": bat,
                                        "end_at": eat,
                                        "slice": frame_x_signal[bat:eat]
                                    })
                                    search_can_end = True
                                    found = True
                                    dec_frames = []
                                    break
                                else:
                                    dec_frames = []
                                    break
                        if found:
                            i = k + 1
                            found = False
                        else:
                            i += 1
                    if not search_can_end and dec_frames_num > dec_frames_num_min:
                        dec_frames_num -= 1
                    elif not search_can_end and dec_frames_num == dec_frames_num_min:
                        search_can_end = True
            bands_fdrs.append({
                "lower_f": band / band_multi,
                "center_f": band,
                "upper_f": band * band_multi,
                "band_energy": band_energy,
                "band_fdrs": band_fdrs
            })
        ret.append({
            "x": frame_x_signal,
            "bands": bands_fdrs
        })
    return ret


def unique_cross_band_fdr_per_superframe(superfrs):
    ret = []
    for superfr in superfrs:
        x = superfr["x"]
        frame = p.DataFrame(superfr["bands"])
        frame["normalized_band_energy"] = frame.band_energy / frame.band_energy.sum()
        filtered = frame[frame.normalized_band_energy > 0.1]
        tmp = []
        for f in filtered.itertuples():
            if len(f.band_fdrs) == 1:
                tmp.append(f.band_fdrs[0])
        if len(tmp) > 0:
            tmp = p.DataFrame(tmp)
            begin_at, end_at = tmp.begin_at.max(), tmp.end_at.max()     # CASE NOISE
            x = x[begin_at:end_at]
            # il filtro mediano a seguire è adoperato per eliminare salt'n'pepper:
            # campioni NaN spuri derivanti dall'algoritmo
            if np.any(np.isnan(x)):
                x = medfilt(x, 3)
            ret.append({
                "begin_at": begin_at,
                "end_at": end_at,
                "slice": x
            })
    return ret


# signal is a mic or band signal of a single mic
# adjustments wrt Capoferri
def entry_point(signal, fs: int):
    signal = me.normalize_signal(signal)
    sub_frame_length = int(0.05 * fs)
    sub_frame_overlap = int(sub_frame_length / 4)
    nfft = int(2 ** np.ceil(np.log2(sub_frame_length)))
    L_lim = int((0.5 * fs) / (sub_frame_length - sub_frame_overlap))

    frame_length = int((2 * fs) / (sub_frame_length - sub_frame_overlap))  # 2 seconds
    frame_overlap = int((2 / 4) * frame_length)

    bands, band_multi = [125, 250, 500, 1000, 2000, 4000], np.sqrt(2)   # oct-bands
    # bands, band_multi = [125, 160, 200, 250,
    #           315, 400, 500, 630,
    #           800, 1000, 1250, 1600,
    #           2000, 2500, 3150, 4000], np.power(2, 1/6)   # third-oct-bands

    bf = find_fdr(signal, fs,
                  frame_length, frame_overlap,
                  sub_frame_length, sub_frame_overlap,
                  bands, band_multi, nfft, L_lim)
    ret = unique_cross_band_fdr_per_superframe(bf)
    return ret
