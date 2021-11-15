import socket
import os


if __name__ == "__main__":
    proj_dir = "/nas/home/gantonacci/Thesis/Project/pythonProject"
    if os.getcwd() != proj_dir:
        os.chdir(proj_dir)
    hostname = socket.gethostname().lower()
    if "bayes" not in hostname:
        # using the free gpu with index 0 or 1 on Euler
        if "euler" in hostname or "gabor" in hostname:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'    # '0' or '1'
        from framework.experimental.nn.usage import nn_utils
        nn_utils.gpus_prepare_memory()

    # sì, gli esperimenti si potrebbero parametrizzare...
    # e sì, sarebbe più estetico avere working directory differenti invece che wrapper per le architetture.

    # <editor-fold desc="RIR CASE">
    # from mains.rlh.rir.nocut.featAG import generation as riraggen
    # riraggen.run()
    # from mains.rlh.rir.nocut.featLeaf import generation as rirlfgen
    # rirlfgen.run()

    # # VOLUME REGRESSION
    # # PER CLASS
    # # \w Genovese
    # from mains.rlh.rir.nocut.featAG.volume_regression import per_class as riragvolregpc
    # riragvolregpc.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rir.nocut.featLeaf.volume_regression import per_class as rirlfvolregpc
    # rirlfvolregpc.run()
    # # PER VOLUME BAND
    # # \w Genovese
    # from mains.rlh.rir.nocut.featAG.volume_regression import per_volume_band as riragvolregvb
    # riragvolregvb.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rir.nocut.featLeaf.volume_regression import per_volume_band as rirlfvolregvb
    # rirlfvolregvb.run()
    # # PER T60 BAND
    # # \w Genovese
    # from mains.rlh.rir.nocut.featAG.volume_regression import per_t60_band as riragvolregtb
    # riragvolregtb.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rir.nocut.featLeaf.volume_regression import per_t60_band as rirlfvolregtb
    # rirlfvolregtb.run()

    # # T60 REGRESSION
    # # WHOLE - TODO andrebbe ri-runnato perché ho cancellato per sbaglio la cartella ahahahah
    # from mains.rlh.rir.nocut.featAG.t60_regression import whole as rev_riragt60regw
    # rev_riragt60regw.run()

    # # SHAPE CLASSIFICATION
    # # WHOLE
    # # \w Genovese
    # from mains.rlh.rir.nocut.featAG.shape_classification import whole as riragshclfw
    # riragshclfw.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rir.nocut.featLeaf.shape_classification import whole as rirlfshclfw
    # rirlfshclfw.run()
    # # PER VOLUME BAND
    # # \w Genovese
    # from mains.rlh.rir.nocut.featAG.shape_classification import per_volume_band as riragshclfvb
    # riragshclfvb.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rir.nocut.featLeaf.shape_classification import per_volume_band as rirlfshclfvb
    # rirlfshclfvb.run()
    # # PER T60 BAND
    # # \w Genovese
    # from mains.rlh.rir.nocut.featAG.shape_classification import per_t60_band as riragshclftb
    # riragshclftb.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rir.nocut.featLeaf.shape_classification import per_t60_band as rirlfshclftb
    # rirlfshclftb.run()
    # # </editor-fold>

    # # <editor-fold desc="REV_WHITE CASE">
    # from mains.rlh.rev_wn.nofdr.featAG import generation as rev_wnaggen
    # rev_wnaggen.run()
    # from mains.rlh.rev_wn.nofdr.featAGAlt import generation as rev_wnagaltgen
    # rev_wnagaltgen.run()
    # from mains.rlh.rev_wn.nofdr.featLeaf import generation as rev_wnlfgen
    # rev_wnlfgen.run()

    # # VOLUME REGRESSION
    # # PER CLASS
    # # \w Genovese
    # from mains.rlh.rev_wn.nofdr.featAG.volume_regression import per_class as rev_wnagvolregpc
    # rev_wnagvolregpc.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rev_wn.nofdr.featLeaf.volume_regression import per_class as rev_wnlfvolregpc
    # rev_wnlfvolregpc.run()
    # # PER VOLUME BAND
    # # \w Genovese
    # from mains.rlh.rev_wn.nofdr.featAG.volume_regression import per_volume_band as rev_wnagvolregvb
    # rev_wnagvolregvb.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rev_wn.nofdr.featLeaf.volume_regression import per_volume_band as rev_wnlfvolregvb
    # rev_wnlfvolregvb.run()
    # # PER T60 BAND
    # # \w Genovese
    # from mains.rlh.rev_wn.nofdr.featAG.volume_regression import per_t60_band as rev_wnagvolregtb
    # rev_wnagvolregtb.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rev_wn.nofdr.featLeaf.volume_regression import per_t60_band as rev_wnlfvolregtb
    # rev_wnlfvolregtb.run()

    # # T60 REGRESSION
    # # WHOLE
    # from mains.rlh.rev_wn.nofdr.featAG.t60_regression import whole as rev_wnagt60regw
    # rev_wnagt60regw.run()

    # # SHAPE CLASSIFICATION
    # # WHOLE
    # # \w Genovese
    # from mains.rlh.rev_wn.nofdr.featAG.shape_classification import whole as rev_wnagshclfw
    # rev_wnagshclfw.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rev_wn.nofdr.featLeaf.shape_classification import whole as rev_wnlfshclfw
    # rev_wnlfshclfw.run()
    # # PER VOLUME BAND
    # # \w Genovese
    # from mains.rlh.rev_wn.nofdr.featAG.shape_classification import per_volume_band as rev_wnagshclfvb
    # rev_wnagshclfvb.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rev_wn.nofdr.featLeaf.shape_classification import per_volume_band as rev_wnlfshclfvb
    # rev_wnlfshclfvb.run()
    # # PER T60 BAND
    # # \w Genovese
    # from mains.rlh.rev_wn.nofdr.featAG.shape_classification import per_t60_band as rev_wnagshclftb
    # rev_wnagshclftb.run()
    # # \w band tb0, \w GenoveseAlt
    # from mains.rlh.rev_wn.nofdr.featAGAlt.shape_classification import per_t60_band as rev_wnagAltshclftb
    # rev_wnagAltshclftb.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rev_wn.nofdr.featLeaf.shape_classification import per_t60_band as rev_wnlfshclftb
    # rev_wnlfshclftb.run()
    # # </editor-fold>

    # # <editor-fold desc="REV_VOICE CASE">
    # from mains.rlh.rev_vc.nofdr.featAG import generation as rev_vcaggen
    # rev_vcaggen.run()
    # from mains.rlh.rev_vc.nofdr.featAGAlt import generation as rev_vcagaltgen
    # rev_vcagaltgen.run()
    # from mains.rlh.rev_vc.nofdr.featLeaf import generation as rev_vclfgen
    # rev_vclfgen.run()

    # # VOLUME REGRESSION
    # # TODO whole neglected, can be easily done similarly to rev_vcagt60regw
    # # PER CLASS
    # # \w Genovese
    # from mains.rlh.rev_vc.nofdr.featAG.volume_regression import per_class as rev_vcagvolregpc
    # rev_vcagvolregpc.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rev_vc.nofdr.featLeaf.volume_regression import per_class as rev_vclfvolregpc
    # rev_vclfvolregpc.run()
    # # PER VOLUME BAND
    # # \w Genovese
    # from mains.rlh.rev_vc.nofdr.featAG.volume_regression import per_volume_band as rev_vcagvolregvb
    # rev_vcagvolregvb.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rev_vc.nofdr.featLeaf.volume_regression import per_volume_band as rev_vclfvolregvb
    # rev_vclfvolregvb.run()
    # # PER T60 BAND
    # # \w Genovese
    # from mains.rlh.rev_vc.nofdr.featAG.volume_regression import per_t60_band as rev_vcagvolregtb
    # rev_vcagvolregtb.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rev_vc.nofdr.featLeaf.volume_regression import per_t60_band as rev_vclfvolregtb
    # rev_vclfvolregtb.run()

    # # T60 REGRESSION
    # # WHOLE
    from mains.rlh.rev_vc.nofdr.featAG.t60_regression import whole as rev_vcagt60regw
    rev_vcagt60regw.run()

    # # SHAPE CLASSIFICATION
    # # WHOLE
    # # \w Genovese
    # from mains.rlh.rev_vc.nofdr.featAG.shape_classification import whole as rev_vcagshclfw
    # rev_vcagshclfw.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rev_vc.nofdr.featLeaf.shape_classification import whole as rev_vclfshclfw
    # rev_vclfshclfw.run()
    # # PER VOLUME BAND
    # # \w Genovese
    # from mains.rlh.rev_vc.nofdr.featAG.shape_classification import per_volume_band as rev_vcagshclfvb
    # rev_vcagshclfvb.run()
    ##############################
    ##############################
    ##############################
    # # \w band vb0, \w GenoveseAlt     # SOLO CASO SCASSATISSIMO - SEMPRE SCASSATO ANCHE COSI'
    # from mains.rlh.rev_vc.nofdr.featAGAlt.shape_classification import per_volume_band as rev_vcagAltshclfvb
    # rev_vcagAltshclfvb.run()
    ##############################
    ##############################
    ##############################
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rev_vc.nofdr.featLeaf.shape_classification import per_volume_band as rev_vclfshclfvb
    # rev_vclfshclfvb.run()
    # # PER T60 BAND
    # # \w Genovese
    # from mains.rlh.rev_vc.nofdr.featAG.shape_classification import per_t60_band as rev_vcagshclftb
    # rev_vcagshclftb.run()
    # # \w band tb0, \w GenoveseAlt
    # from mains.rlh.rev_vc.nofdr.featAGAlt.shape_classification import per_t60_band as rev_vcagAltshclftb
    # rev_vcagAltshclftb.run()
    # # \w Leaf'n'ConvNET
    # # TODO go with the secondary project in src/refactors/leaf-audio
    # from mains.rlh.rev_vc.nofdr.featLeaf.shape_classification import per_t60_band as rev_vclfshclftb
    # rev_vclfshclftb.run()
    # # </editor-fold>

    print("EXIT")
