from keras.models import Model
from pathlib import Path
import os
from keras.utils import Sequence
import pandas as p
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from eg.eg_nn import predict_classif_model, predict_regress_model, load_model_history
from eg.reports import eg_reports_feat_maps, eg_reports_voice_acquisitions
from framework.data import fs_utils
import framework.experimental.nn.usage.batch_generation2 as bg2


def __get_model_report_dir(base_model_dir: str) -> str:
    return "{}/!report".format(base_model_dir)


def __get_model_attention_dir(base_model_dir: str) -> str:
    return "{}/attention".format(__get_model_report_dir(base_model_dir))


def get_model_report_dir(base_model_dir: str) -> str:
    report_dir = __get_model_report_dir(base_model_dir)
    fs_utils.dir_exists_or_create(report_dir)
    return report_dir


def get_model_attention_dir(base_model_dir: str) -> str:
    attention_dir = __get_model_attention_dir(base_model_dir)
    fs_utils.dir_exists_or_create(attention_dir)
    return attention_dir


def __create_model_report(test_model: Model, report_dir: str):
    plot_model(test_model, to_file="{}/architecture.png".format(report_dir),
               show_shapes=True, show_layer_names=True)
    return


def __create_train_val_report(v_train, v_valid, highlight: str, yname: str, report_dir: str, image_name: str):
    assert highlight in ["min", "max", "ntz"]
    argf = None
    if highlight == "max":
        argf = np.argmax
    else:
        argf = np.argmin
    arg = v_valid
    # if highlight == "ntz":
    #     arg = np.abs(v_valid)
    # else:
    #     arg = v_valid

    epo_arg, v_val = argf(arg) + 1, v_valid[argf(arg)]
    epochs = range(1, len(v_train) + 1)

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$N_{EPOCH}$")
    ax.set_ylabel(r"${}$".format(yname))
    ax.yaxis.set_label_position("right")
    plt.plot(epochs, v_train, color="red", label="Training")
    plt.plot(epochs, v_valid, color="green", label="Validation")
    plt.scatter(epo_arg, v_val, color="blue", label=r"$Validation_{"+highlight+"}$")
    ax.annotate("x = {}\ny = {:.3f}".format(epo_arg, v_val), xy=(epo_arg, v_val), color="orange")
    plt.legend()
    plt.grid()
    plt.savefig("{}/{}.png".format(report_dir, image_name), dpi=300)
    plt.close(fig)
    return


def __create_lr_report(lr, report_dir):
    epochs = range(1, len(lr) + 1)

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$N_{EPOCH}$")
    ax.set_ylabel(r"$log_{10}(LR)$")
    plt.plot(epochs, np.log10(lr), color="red", label="Training")
    plt.legend()
    plt.grid()
    plt.savefig("{}/lr.png".format(report_dir), dpi=300)
    plt.close(fig)
    return


def confusion_matrix(ygt, ypred):
    return tf.math.confusion_matrix(ygt, ypred)


def normalize_confusion_matrix(mat, num_classes):
    return mat / np.repeat(np.sum(mat, axis=1).reshape((num_classes, 1)), num_classes, axis=1)


def __create_confusion_report(ygt, ypred, classes: [str], report_dir: str):
    fig, ax = plt.subplots()
    lin = np.arange(len(classes))
    # for 2 classes: [[TN, FP], [FN, TP]], r is GT (N, P), c is PRED (N, P)
    # reference: https://stackoverflow.com/questions/52852742/how-to-read-tensorflow-confusion-matrix-rows-and-columns
    mat = confusion_matrix(ygt, ypred)
    # row-wise normalization
    mat = normalize_confusion_matrix(mat, len(classes))

    ax.set_xlabel(r"$y_{PRED}$")
    ax.set_ylabel(r"$y_{GT}$")
    ax.yaxis.set_label_position("right")
    ax.set_xticks(lin)
    ax.set_yticks(lin)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.matshow(mat, cmap=plt.get_cmap("summer"))

    for i in lin:
        for j in lin:
            # ax.text(c, r, ... mat[r, c] ...)
            ax.text(j, i, "{:.2f}".format(mat[i, j]), va='center', ha='center', size="x-large")
    plt.tight_layout()
    plt.savefig("{}/confusion.png".format(report_dir), dpi=300)
    plt.close(fig)
    return


def __create_prediction_report(ygt, ypred, report_dir: str, varname: str, udm: str):
    fig = plt.figure()
    plt.plot(range(len(ygt)), ygt, color="r", label=r"$"+varname+"_{GT}$")
    plt.plot(range(len(ypred)), ypred, color="g", label=r"$"+varname+"_{PRED}$")
    plt.xlim([100, 200])
    plt.xlabel(r"$N_{sample}$")
    plt.ylabel(r"${} [{}]$".format(varname, udm))
    plt.legend()
    plt.savefig("{}/prediction.png".format(report_dir))
    plt.close(fig)
    return

# NB the implementation has been updated
def __make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    import keras
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def __create_gradcam_report(rev_acquisition, fmap, heatmap, feature_names: [str], out_shape, attention_dir, i):
    import matplotlib.cm as cm
    import keras
    fmap = np.uint8(255 * fmap)
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :fmap.shape[-1]]
    hmap = jet_colors[heatmap]

    # resize
    fmapim = keras.preprocessing.image.array_to_img(fmap)
    fmapim = fmapim.resize(out_shape)
    fmap = keras.preprocessing.image.img_to_array(fmapim)

    hmapim = keras.preprocessing.image.array_to_img(hmap)
    hmapim = hmapim.resize(out_shape)
    hmap = keras.preprocessing.image.img_to_array(hmapim)

    # Superimpose the heatmap on original image
    superposed = hmap * 0.4 + fmap
    superposedim = keras.preprocessing.image.array_to_img(superposed)

    # QUI USARE MATPLOTLIB RALLENTA TUTTO :(
    # eg_reports_noise_acquisitions e eg_reports_voice_acquisitions sono quasi del tutto simili,
    # schiaffo create_basic in uno dei due a caso, si dovrebbe crear un solo file più generico, ma anche basta
    # POTA
    revfig = eg_reports_voice_acquisitions.create_basic(rev_acquisition)
    fmapfig = eg_reports_feat_maps.inframe(fmap[:, :, 0], feature_names)
    hmapfig = eg_reports_feat_maps.inframe(hmap[:, :, 0], feature_names)
    superposedfig = eg_reports_feat_maps.inframe(superposed[:, :, 0], feature_names)

    # Save the superimposed image
    # TODO STAMPARE L'ACQUISIZIONE ASSOCIATA
    revfig.savefig("{}/sample_{}_rev.png".format(attention_dir, i))
    fmapfig.savefig("{}/sample_{}_fmap.png".format(attention_dir, i))
    hmapfig.savefig("{}/sample_{}_hmap.png".format(attention_dir, i))
    superposedfig.savefig("{}/sample_{}_superposed.png".format(attention_dir, i))

    plt.close(revfig)
    plt.close(fmapfig)
    plt.close(hmapfig)
    plt.close(superposedfig)
    return


def create_gradcam_reports(base_model_dir: str,
                            test_gen: bg2.BatchGenerator2, test_model: Model,
                            # where to extract feat_map, lin layers
                            last_conv_layer_name: str, classifier_layer_names: [str],
                            feature_names: [str]):
    from framework.data.io_relations.dirs import file_pre_feat_df_chain
    attention_dir = get_model_attention_dir(base_model_dir)

    featmap_paths = [test_gen.subset_batch(0).input_file_path.values[i] for i in range(20)]
    featmap_assoc_sig = [file_pre_feat_df_chain.get_pre_feat_df_by_chaining(fmappath).output.values[0].reshape((-1))
                         for fmappath in featmap_paths]
    imgs_hwc = [test_gen[0][0][i] for i in range(20)]     # batch 0, X, i
    imgs_heat = [__make_gradcam_heatmap(np.expand_dims(img_hwc, axis=0), test_model, last_conv_layer_name, classifier_layer_names)
                 for img_hwc in imgs_hwc]
    for i in range(len(imgs_hwc)):
        __create_gradcam_report(featmap_assoc_sig[i], imgs_hwc[i], imgs_heat[i], feature_names, (1080, 1080), attention_dir, i)
    return


def create_classif_report(base_model_dir: str, test_model: Model, test_gen: Sequence, classes: [str],
                          model_report: bool = True, history_report: bool = True, confusion_report: bool = True):
    report_dir = get_model_report_dir(base_model_dir)

    if model_report:
        __create_model_report(test_model, report_dir)
    if history_report:
        h: p.DataFrame = load_model_history(base_model_dir)
        __create_lr_report(h.lr.values, report_dir)
        __create_train_val_report(h.loss.values, h.val_loss.values,
                             "min", "H_{y_{GT}, y_{PRED}}", report_dir, "categorical_crossentropy")

        __create_train_val_report(h.accuracy.values, h.val_accuracy.values,
                             "max", "ACC_{y_{GT}, y_{PRED}}", report_dir, "accuracy")
    if confusion_report:
        ygt, ypred = predict_classif_model(test_model, test_gen)
        __create_confusion_report(ygt, ypred, classes, report_dir)
    return


def create_regress_report(base_model_dir: str, test_model: Model, test_gen: Sequence,
                          model_report: bool = True, history_report: bool = True, prediction_report: bool = True,
                          varname: str = "V", udm: str = "m^3"):
    report_dir = get_model_report_dir(base_model_dir)

    if model_report:
        __create_model_report(test_model, report_dir)
    if history_report:
        h: p.DataFrame = load_model_history(base_model_dir)
        __create_lr_report(h.lr.values, report_dir)
        __create_train_val_report(h.loss.values, h.val_loss.values,
                             "min", "MSE_{y_{GT}, y_{PRED}}", report_dir, "mse")

        __create_train_val_report(h.pearsons_coeff.values, h.val_pearsons_coeff.values,
                             "max", "P_{y_{GT}, y_{PRED}}", report_dir, "pearsons_coeff")
        __create_train_val_report(h.mean_absolute_error.values, h.val_mean_absolute_error.values,
                             "min", "MAE_{y_{GT}, y_{PRED}}", report_dir, "mean_absolute_error")
        __create_train_val_report(h.mean_mult.values, h.val_mean_mult.values,
                             "min", "MM_{y_{GT}, y_{PRED}}", report_dir, "mean_mult")
    if prediction_report:
        ygt, ypred = predict_regress_model(test_model, test_gen)
        __create_prediction_report(ygt, ypred, report_dir, varname, udm)
    return


def __crvr_var_distro_comparison(y: p.DataFrame, split_name: str, report_dir: str, varname: str, udm: str):
    from framework.experimental.nn import dataset_utilities as dsu
    fig, axs = plt.subplots(2, 1)
    m = 0   # np.min([y.gnd.min(), y.pred.min()])
    M = np.max([y.gnd.max(), y.pred.max()])
    for ax in axs:
        ax.set_xlabel(r"${} [{}]$".format(varname, udm))
        ax.set_xlim([m, M])
    axs[0].set_ylabel(r"$c("+varname+"_{GT})$")
    axs[1].set_ylabel(r"$c("+varname+"_{PRED})$")
    (vgt, fgt), (vpred, fpred) = dsu.get_field_frequencies(y, "gnd"), dsu.get_field_frequencies(y, "pred")
    axs[0].bar(vgt, fgt)
    # il secondo plot (sotto) è normale che sia abbastanza uniforme perchè la predizione
    # genera scalari simili al gt, ma del tutto unici
    axs[1].bar(vpred, fpred)
    plt.savefig("{}/{}_distro_{}_comparison.png".format(report_dir, varname, split_name))
    plt.close(fig)
    return


def __crvr_var_bands_comparison(y: p.DataFrame, split_name: str, report_dir: str, varname: str, udm: str):
    fig = plt.figure()
    m = 0   # np.min([y.gnd.min(), y.pred.min()])
    M = np.max([y.gnd.max(), y.pred.max()])
    y.plot.hexbin(x="gnd", y="pred", gridsize=20,
                  extent=(m, M, m, M),
                  xlabel=r"$"+varname+"_{GT} ["+udm+"]$",
                  ylabel=r"$"+varname+"_{PRED} ["+udm+"]$")
    plt.savefig("{}/{}_band_hexbin_{}.png".format(report_dir, varname, split_name))
    plt.close(fig)
    return


def __create_regress_var_report(ygt, ypred, split_name: str, report_dir: str, varname: str, udm: str):
    y = p.DataFrame([ygt, ypred]).T
    y.columns = ["gnd", "pred"]

    # siano a, b = V_{GT}, V_{PRED}
    # stampare p(a | split_name) e p(b | split_name)
    __crvr_var_distro_comparison(y, split_name, report_dir, varname, udm)

    # siano a', b' raggruppamenti su a, b rispettivamente
    # stampare hexbin
    __crvr_var_bands_comparison(y, split_name, report_dir, varname, udm)
    return


def create_regress_var_reports(base_model_dir: str, test_model: Model, splits: [{str, Sequence}],
                               varname: str = "V", udm: str = "m^3"):
    report_dir = get_model_report_dir(base_model_dir)

    for split in splits:
        ygt, ypred = predict_regress_model(test_model, split["generator"])
        __create_regress_var_report(ygt, ypred, split["split_name"], report_dir, varname, udm)
    return
