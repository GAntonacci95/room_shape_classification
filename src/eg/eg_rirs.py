import functools

import pandas as p

import framework.experimental.simulation.room_utilities as ru
from pyroomacoustics import Room
from framework.data.thread_timing import ThreadTimer

from framework.data.io_relations.dirs import one_to_one_or_many
import numpy as np


def singlerirgen(setup_frame, material_arg, f_s, max_ord, use_ray):
    # import matplotlib.pyplot as plt
    # from framework.extension.scipy import hp_filter
    praroom: Room = None
    t60_measured = -1
    try:
        praroom = ru.praroom_from_setup_df(setup_frame, material_arg, f_s, max_ord, use_ray)
        # # PRINT IN CASE OF DBG PURPOSES
        # fig, ax = praroom.plot()
        # fig.show()
        praroom.compute_rir()

        # praroom.plot_rir([(0, 0)])
        # plt.show()

        # for i in range(len(praroom.rir)):   # for src
        #     for j in range(len(praroom.rir[i])):    # for mic
        #         pre = praroom.rir[i][j]
        #         # # PRINT IN CASE OF DBG PURPOSES
        #         # plt.plot(range(len(pre)), pre)
        #         # plt.show()
        #         # plt.clf()
        #         post = hp_filter(pre.reshape((1, -1)), 130, setup_frame.f_s.values[0]).reshape(-1)
        #         praroom.rir[i][j] = post
        #         # # PRINT IN CASE OF DBG PURPOSES
        #         # plt.plot(range(len(post)), post)
        #         # plt.show()
        #         # plt.clf()

        # NB: opportuno normalizzare le acquisizioni!
    except Exception as e:  # ho rispettato la notazione, se qualcosa non va debug!
        print("RirsGen: U BETTER DEBUG")

    try:
        t60_measured = praroom.measure_rt60()
    except Exception as e:  # exception catch and handling
        print("RirsGen t60: U BETTER DEBUG")
        t60_measured = np.array([[0.0]])
    return praroom, t60_measured


# se des - meas > 0 (grad +), meas < des, voglio meas -> des, devo ridurre l'assorbimento: diff = - * +
# se des - meas < 0 (grad -), meas > des, voglio meas -> des, devo aumentare l'assorbimento: diff = - * -
def gradient_descent(gradient, start, lr=0.1, n_iter=50, tolerance=1e-06):
    vector = start
    for _ in range(n_iter):
        deltat = gradient(vector)
        # non conoscendo il modo, qua tengo pere (assorb) con mele -µ(∆t60):
        # e la tolleranza e l'aggiornamento sono magici
        diff = -lr * deltat
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector


def mygrad(material_arg_i, setup_frame, t60_desired, f_s, max_ord, use_ray):
    praroom, t60_measured = singlerirgen(setup_frame, material_arg_i, f_s, max_ord, use_ray)
    # il grad ∆t andrebbe in qualche modo rimappato su una variazione di assorbimento...
    # non credo di conoscere il modo...
    return t60_desired - t60_measured[0][0]


def rirs(df: p.DataFrame, params: dict) -> None:
    print("RIRs generation")
    tt = ThreadTimer()
    if (("wall_absorptions" in params.keys()) == ("v_range" in params.keys())):
        raise Exception("RirsGen: wall_absorptions xor (v_range and t60_range) params must be provided.")
    f_s = params["f_s"]
    max_ord = params["max_ord"]
    use_ray = params["use_ray"]
    t60th, lr = 0.05, 0.04004
    # if use_ray and max_ord != 3:
    #     raise Exception("RirsGen: pra usage of ray_tracing is unstable. max_ord=3 shall be used.")

    for t in df.itertuples():
        tt.start()

        setup_frame = p.read_pickle(t.input_file_path)
        georoom = ru.georoom_from_setup_df(setup_frame)
        t60_desired = -1
        material_arg = None
        if "v_range" in params.keys():
            v_range, t60_range = params["v_range"], params["t60_range"]
            t60_desired = ru.ignorant_v_to_t60_map(v_range, t60_range, georoom.volume)
            material_arg = ru.e_absorption(georoom, t60_desired)
        else: # given xor, can be only walls_absorptions
            wall_absorptions = params["wall_absorptions"]
            material_arg = wall_absorptions[0] if len(wall_absorptions) == 1 else \
                    np.random.choice(wall_absorptions, 1, replace=False)[0]

        # SETUP & COMPUTE_RIR
        if "v_range" in params.keys():
            # while abs(t60_desired - t60_measured[0][0]) > t60th:
            #     measgtdes = t60_measured[0][0] > t60_desired
            #     material_arg = material_arg + material_var if measgtdes else material_arg - material_var
            #     praroom, t60_measured = singlerirgen(setup_frame, material_arg, f_s, max_ord, use_ray)
            #     updated_measgtdes = t60_measured[0][0] > t60_desired
            #     if measgtdes != updated_measgtdes:
            #         material_var -= material_var_red
            material_arg = gradient_descent(
                gradient=functools.partial(mygrad,
                                           setup_frame=setup_frame, t60_desired=t60_desired,
                                           f_s=f_s, max_ord=max_ord, use_ray=use_ray),
                start=material_arg, lr=lr, tolerance=lr*t60th
            )
            praroom, t60_measured = singlerirgen(setup_frame, material_arg, f_s, max_ord, use_ray)
        else:
            praroom, t60_measured = singlerirgen(setup_frame, material_arg, f_s, max_ord, use_ray)

        datum = {
            "input_file_path": t.input_file_path,  # given the setup, the resulting rir is
            "material_arg": material_arg,
            "f_s": f_s,
            "max_ord": max_ord,
            "use_ray": use_ray,
            "rir": praroom.rir,  # lista tipo h(m, s, t) dove per costruzione dei setup m = 1
            "t60": t60_measured          # lista tipo t60(m, s) dove per costruzione dei setup m = 1
        }
        # SALVATAGGIO, l'hashing qui non è necessario perchè 1 setup -> 1 rir
        p.DataFrame([datum]).to_pickle("{}/{}.pkl".format(t.output_dir, t.input_file_name))

        tt.end_first(df.shape[0])

    tt.end_total()
    return


def compute(egsetups_output_dir: str, params: dict) -> None:
    one_to_one_or_many.compute_parallel(
        egsetups_output_dir,
        {"handler": rirs, "params": params},
        "./datasets/rooms/setups"
    )
    return


def load(egsetups_output_dir: str, params: dict) -> str:
    return one_to_one_or_many.get_output_dir(
        egsetups_output_dir,
        {"handler": rirs, "params": params},
        "./datasets/rooms/setups"
    )
