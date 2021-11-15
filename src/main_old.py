from mains_old import main_rlh_rir_ismplus_nocut_generation
    # main_rlh_ismray_wn_nofdr_for_leaf_generation
    # main_rlh_ismray_vc_nofdr_for_leaf_generation
    # main_rlh_ismray_vc_nofdr_featAG_netGA
    # main_rlh_ismray_wn_nofdr_featAG_netGA
    # main_rlh_ismray_vc_nofdr_featAG_netAG
    # main_rlh_ismray_vc_nofdr_featAG_generation
    # main_rlh_ismray_wn_nofdr_featAG_netAG
    # main_rlh_ismray_wn_nofdr_featAG_generation
    # main_rirsirmray_vc_nofdr_featAG_generation
    # main_absorption_investigation
    # main_rlh_rir_ismplus_nocut_generation
    # main_rlh_rir_nocut_generation
    # main_ccrirs_inspection
    # main_rlh_rir_ismplus_time_consumption
    # main_rlh_wn_frmp0_nofdr_for_leaf_generation
    # main_vc_frmp0_fdrDC_for_leaf_generation
    # main_rir_nocut_generation
    # main_rir_cut_generation
    # main_rir_nocut_netMLP
    # main_rir_cut_netMLP
    # main_rir_cut_nofeat_RF
    # main_rlh_rir_nocut_netMLP
    # main_wn_frmp0_nofdr_featAG_generation
    # main_wn_frmp0_fdrDC_featAG_generation
    # main_wn_frmp0_fdrCC_featAG_generation
    # main_rlh_wn_frmp0_nofdr_featAG_generation
    # main_wn_frmp0_nofdr_featAG_netAG
    # main_wn_frmp0_nofdr_featAG_volRestr_netAG
    # main_rlh_wn_frmp0_nofdr_featAG_netAG
    # main_wn_frmp0_nofdr_featAG_netGA
    # main_wn_frmp0_nofdr_featAG_netXception
    # main_wn_frmp0_nofdr_featAG_volRestr_netGA
    # main_wn_frmp0_nofdr_featAG_volRestr_netXception
    # main_rlh_wn_frmp0_nofdr_featAG_netGA
    # main_vc_frmp0_nofdr_featAG_generation
    # main_vc_frmp0_nofdr_for_leaf_generation
    # main_vc_frmp0_fdrDC_featAG_generation
    # main_vc_frmp0_fdrCC_featAG_generation
    # main_vc_frmp0_fdrCC_for_leaf_generation
    # main_vc_frmp0_nofdr_featAG_netGA
    # main_vc_frmp0_fdrCC_featAG_netGA
    # main_vc_frmp0_fdrCC_featAG_netXception
import os
import socket


if __name__ == "__main__":
    hostname = socket.gethostname().lower()
    if "bayes" not in hostname:
        # using the free gpu with index 0 or 1 on Euler
        if "euler" in hostname:
            os.environ["CUDA_VISIBLE_DEVICES"] = '1'    # '0' or '1'
        from framework.experimental.nn.usage import nn_utils
        nn_utils.gpus_prepare_memory()

    # # RIR CASE    # TODO redo report setups e rirs
    # main_rir_nocut_generation.run()       # VBB
    # main_rir_cut_generation.run()         # VBB
    # main_rlh_rir_nocut_generation.run()           # NI
    # main_ccrirs_inspection.run()
    # main_absorption_investigation.run()   # se non dovesse andare la distro dei setup rivedere qui
    # main_rlh_rir_ismplus_time_consumption.run()
    # main_rlh_rir_ismplus_nocut_generation.run()   # NI, ma le rir sono più lunghe e la distro sui T60 è più bella =)

    # # SHAPE CLASSIFICATION
    # main_rir_nocut_netMLP.run()           # REM
    # main_rir_cut_netMLP.run()             # REM
    # main_rir_cut_nofeat_RF.run()          # REM
    # main_rlh_rir_nocut_netMLP.run()       # OK TIL ISM+RAY?

    # # REVERBERANT WHITE NOISE CASE    #   TODO: DOVREI RIFARE SIMs, MA QUI HO QUALCHE RISULTATO.
    # main_wn_frmp0_nofdr_featAG_generation.run()       # OK
    # main_wn_frmp0_fdrDC_featAG_generation.run()       # FDR SPACCATE - INUTILE
    # main_wn_frmp0_fdrCC_featAG_generation.run()       # FDR SPACCATE - INUTILE
    # main_rlh_wn_frmp0_nofdr_featAG_generation.run()   # CANCELLARE SOPRA - OK - TODO REDO RIR NUOVE
    # main_rlh_wn_frmp0_nofdr_for_leaf_generation.run()
    # main_rlh_ismray_wn_nofdr_for_leaf_generation.run()
    # main_rlh_ismray_wn_nofdr_featAG_generation.run()    # DEFINITIVO? TODO MAPS REPORT
    # # R/L VOLUME REGRESSION    # TODO: BATCH NORM
    # main_wn_frmp0_nofdr_featAG_netAG.run()            # VBB
    # main_wn_frmp0_nofdr_featAG_volRestr_netAG.run()   # VBB
    # main_rlh_wn_frmp0_nofdr_featAG_netAG.run()        # RIFO ANCORA DOPO CLASSIF # TODO REDO LATER, DUE TO RIR and BN - min
    # main_rlh_ismray_wn_nofdr_featAG_netAG.run()        # DEFINITIVO?
    # # SHAPE CLASSIFICATION
    # main_wn_frmp0_nofdr_featAG_netGA.run()                   # BUONINO, dati i risultati RIR
    # main_wn_frmp0_nofdr_featAG_netXception.run()             # BUONINO, dati i risultati RIR
    # main_wn_frmp0_nofdr_featAG_volRestr_netGA.run()          # BUONINO, dati i risultati RIR
    # main_wn_frmp0_nofdr_featAG_volRestr_netXception.run()    # BUONINO, dati i risultati RIR
    # main_rlh_wn_frmp0_nofdr_featAG_netGA.run()               # BUONINO, dati i risultati RIR, OK, TODO REDO REPORT
    # main_rlh_ismray_wn_nofdr_featAG_netGA.run()               # DEFINITIVO?

    # REVERBERANT VOICE CASE, ci vorrebbe la sorgente rumorosa aggiuntiva, problema sampling corona sferica
    # main_vc_frmp0_nofdr_featAG_generation.run()     # OK
    # main_vc_frmp0_nofdr_for_leaf_generation.run()   # NICE RESULTS
    # main_vc_frmp0_fdrDC_featAG_generation.run()     # SHEET
    # main_vc_frmp0_fdrDC_for_leaf_generation.run()   # TODO HERE
    # main_vc_frmp0_fdrCC_featAG_generation.run()     # FDR ACCETTABILI - MA SHEET PER TRAIN GA/LEAF
    # main_vc_frmp0_fdrCC_for_leaf_generation.run()   # SHEET
    # main_rlh_ismray_vc_nofdr_for_leaf_generation.run()
    # main_rlh_ismray_vc_nofdr_featAG_generation.run()    # DEFINITIVO? TODO MAPS REPORT
    # VOLUME REGRESSION
    # main_rlh_ismray_vc_nofdr_featAG_netAG.run()        # DEFINITIVO?
    # # SHAPE CLASSIFICATION
    # main_vc_frmp0_nofdr_featAG_netGA.run()          # RISULTATI BRUTTI - OK
    # main_vc_frmp0_fdrCC_featAG_netGA.run()          # RISULTATI BRUTTI - OK
    # main_vc_frmp0_fdrCC_featAG_netXception.run()    # RISULTATI BRUTTI - FATTIBILE, MA NON VA LONTANO
    # main_rlh_ismray_vc_nofdr_featAG_netGA.run()               # SEMPRE SCASSATO

    print("EXIT")
