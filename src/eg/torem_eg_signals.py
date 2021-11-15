# import pyroomacoustics as pra
# import framework.experimental.simulation.rooms.signals as rs
# import pandas as p
#
#
# def gen_sigs(irss: p.DataFrame, irs_fs: int, input_base_dir: str):
#     return rs.parallel_simulate(
#         irss=irss,
#         irs_fs=irs_fs,
#         input_base_dir=input_base_dir
#     )
#
#
# def dump_sigs(df: p.DataFrame, output_file_dir: str, f_s: int, max_ord: int, n_further_rnd_mic_pos: int = 0):
#     df.to_pickle("{}/eg_fs{}_ord{}_frmp{}.pkl".format(
#         output_file_dir,
#         f_s,
#         max_ord,
#         n_further_rnd_mic_pos
#     ))
#     return
#
#
# def load_sigs(output_file_dir: str, f_s: int, max_ord: int, n_further_rnd_mic_pos: int = 0):
#     return p.read_pickle("{}/eg_fs{}_ord{}_frmp{}.pkl".format(
#         output_file_dir,
#         f_s,
#         max_ord,
#         n_further_rnd_mic_pos
#     ))
#
#
# def listen_sample(df: p.DataFrame, f_s: int):
#     raise Exception("Not implemented yet!")
#     return
