# import framework.extension.math as me
# from obspy.signal.filter import envelope
# from scipy.ndimage.filters import uniform_filter1d
# from framework.experimental.nn.features.pre_processing import nnpreprocutils
# from librosa.feature import melspectrogram
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def find_fdrs(grad):
#     a1, b1 = grad >= 0.4, np.gradient(grad) > 0
#     a2, b2 = grad <= -0.4, np.gradient(grad) < 0
#     spoken_begins = np.argwhere(a1 & b1)
#     spoken_ends = np.argwhere(a2 & b2)
#     fdrs_begins = None
#     fdrs_ends = None
#     return zip(fdrs_begins, fdrs_ends)
#
#
# def entry_point(signal, f_s):
#     ret = []
#     signal = me.normalize_signal(signal)
#     spec = melspectrogram(signal, f_s/2, win_length=512, hop_length=256, n_mels=16)
#
#     fig, (ax1, ax2) = plt.subplots(2, 1)
#     ax1.plot(range(len(signal)), signal)
#     ax2.imshow(spec)
#     plt.show()
#     plt.close(fig)
#
#     # plt.figure()
#     # plt.plot(signal, color="yellow")
#     # plt.plot(env, color="green")
#     # plt.plot(incipit, color="purple")
#     # plt.plot(grad, color="blue")
#     #
#     # for begin_at, end_at in final_fdrs:
#     #     ret.append({        # DBG
#     #         "begin_at": begin_at,
#     #         "end_at": end_at,
#     #         "slice": signal[begin_at:end_at]
#     #     })
#     #     plt.axvspan(begin_at, end_at, color="red", alpha=0.3)
#     #
#     # plt.grid()
#     # plt.show()
#
#     return ret      # DBG
