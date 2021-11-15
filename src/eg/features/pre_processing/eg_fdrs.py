import pandas as p
from framework.data.thread_timing import ThreadTimer
import numpy as np
from framework.data.io_relations.dirs import one_to_one_or_many

# TODO: NB NON C'E' UN ALGORITMO CHE SIA STABILE PER L'ESTRAZIONE DELLE FDR!
#  non avendo tempo per risolvere sfrutto l'algoritmo di Capoferri sulle voci
#  anche se nessun algoritmo funziona sul rumore riverberante

# trying a solution based on ccastelnuovo
import refactors.ccastelnuovo.pre_processing.fdrs as ref_fdrs_cc
# trying a solution based on dcapoferri
import refactors.dcapoferri.pre_processing.fdrs as ref_fdrs_dc
# trying a solution based on hmalik
import refactors.hmalik.pre_processing.fdrs as ref_fdrs_hm
# personal trial
import framework.experimental.nn.features.pre_processing.fdrs as ref_fdrs_ga


def fdrsCC(df: p.DataFrame, params: dict) -> None:
    print("fdrCCs generation")
    f_s = params["f_s"]
    lmin, lmax = params["len_min_samps"], params["len_max_samps"]
    tt = ThreadTimer()
    for t in df.itertuples():
        tt.start()
        # il file caricato è o il segnale di un microfono
        signal = p.read_pickle(t.input_file_path).output.values[0]
        # per esso CC fornisce [0, N] FDR
        file_fdrs = ref_fdrs_cc.entry_point(np.array(signal, dtype=np.float), f_s)
        for i in range(len(file_fdrs)):
            if lmin <= file_fdrs[i]["end_at"] - file_fdrs[i]["begin_at"] + 1 <= lmax:
                datum = {
                    "input_file_path": t.input_file_path,
                    "begin_at": file_fdrs[i]["begin_at"],
                    "end_at": file_fdrs[i]["end_at"],
                    "output": file_fdrs[i]["slice"]
                }
                p.DataFrame([datum]).to_pickle("{}/{}_{}.pkl".format(t.output_dir, t.input_file_name, i))
        tt.end_first(df.shape[0])
    tt.end_total()
    return


def computeCC(acqus_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        acqus_output_dir,
        {"handler": fdrsCC, "params": params},
        "./datasets/nn/features/pre_processing"
    )
    return


def loadCC(acqus_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        acqus_output_dir,
        {"handler": fdrsCC, "params": params},
        "./datasets/nn/features/pre_processing"
    )


def fdrsDC(df: p.DataFrame, params: dict) -> None:
    print("fdrDCs generation")
    f_s = params["f_s"]
    lmin, lmax = params["len_min_samps"], params["len_max_samps"]
    tt = ThreadTimer()
    for t in df.itertuples():
        tt.start()
        # il file caricato è o il segnale di un microfono
        signal = p.read_pickle(t.input_file_path).output.values[0]
        file_fdrs = ref_fdrs_dc.entry_point(np.array(signal, dtype=np.float), f_s)
        for i in range(len(file_fdrs)):
            if lmin <= file_fdrs[i]["end_at"] - file_fdrs[i]["begin_at"] + 1 <= lmax:
                datum = {
                    "input_file_path": t.input_file_path,
                    "begin_at": file_fdrs[i]["begin_at"],
                    "end_at": file_fdrs[i]["end_at"],
                    "output": file_fdrs[i]["slice"]
                }
                p.DataFrame([datum]).to_pickle("{}/{}_{}.pkl".format(t.output_dir, t.input_file_name, i))
        tt.end_first(df.shape[0])
    tt.end_total()
    return


def computeDC(acqus_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        acqus_output_dir,
        {"handler": fdrsDC, "params": params},
        "./datasets/nn/features/pre_processing"
    )
    return


def loadDC(acqus_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        acqus_output_dir,
        {"handler": fdrsDC, "params": params},
        "./datasets/nn/features/pre_processing"
    )
#
#
# def fdrsHM(df: p.DataFrame, params: dict) -> None:
#     f_s = params["f_s"]
#     print("fdrHMs generation")
#     tt = ThreadTimer()
#     for t in df.itertuples():
#         tt.start()
#         # il file caricato è o il segnale di un microfono
#         signal = p.read_pickle(t.input_file_path).output.values[0]
#         file_fdrs = ref_fdrs_hm.entry_point(np.array(signal, dtype=np.float), f_s)
#         for i in range(len(file_fdrs)):
#             datum = {
#                 "input_file_path": t.input_file_path,
#                 # I DETTAGLI DI INIZIO E FINE SONO INCLUSI
#                 "output": file_fdrs[i]
#             }
#             # p.DataFrame([datum]).to_pickle("{}/{}_{}.pkl".format(t.output_dir, t.input_file_name, i))
#         tt.end_first(df.shape[0])
#     tt.end_total()
#     return
#
# # TODO: DA SEQ A PAR
# #   NB, IMPLEMENTAZIONE INCOMPLETA
# def computeHM(acqus_output_dir: str, params: dict) -> None:
#     raise Exception("UNSTABLE")
#     one_to_one_or_many.compute_sequential(
#         acqus_output_dir,
#         {"handler": fdrsHM, "params": params},
#         "./datasets/nn/features/pre_processing"
#     )
#     return
#
#
# def loadHM(acqus_output_dir: str, params: dict) -> str:
#     return one_to_one_or_many.get_output_dir(
#         acqus_output_dir,
#         {"handler": fdrsHM, "params": params},
#         "./datasets/nn/features/pre_processing"
#     )
#
#
# def fdrsGA(df: p.DataFrame, params: dict) -> None:
#     f_s = params["f_s"]
#     print("fdrGAs generation")
#     tt = ThreadTimer()
#     for t in df.itertuples():
#         tt.start()
#         # il file caricato è o il segnale di un microfono
#         signal = p.read_pickle(t.input_file_path).output.values[0]
#         file_fdrs = ref_fdrs_ga.entry_point(np.array(signal, dtype=np.float), f_s)
#         for i in range(len(file_fdrs)):
#             datum = {
#                 "input_file_path": t.input_file_path,
#                 "begin_at": file_fdrs[i]["begin_at"],
#                 "end_at": file_fdrs[i]["end_at"],
#                 "output": file_fdrs[i]["slice"]
#             }
#             # p.DataFrame([datum]).to_pickle("{}/{}_{}.pkl".format(t.output_dir, t.input_file_name, i))
#         tt.end_first(df.shape[0])
#     tt.end_total()
#     return
#
#
# def computeGA(acqus_output_dir: str, params: dict) -> None:
#     raise Exception("INCOMPLETE")
#     one_to_one_or_many.compute_sequential(
#         acqus_output_dir,
#         {"handler": fdrsGA, "params": params},
#         "./datasets/nn/features/pre_processing"
#     )
#     return
#
#
# def loadGA(acqus_output_dir: str, params: dict) -> str:
#     return one_to_one_or_many.get_output_dir(
#         acqus_output_dir,
#         {"handler": fdrsGA, "params": params},
#         "./datasets/nn/features/pre_processing"
#     )
