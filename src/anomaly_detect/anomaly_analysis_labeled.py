import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
import yaml

from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

from src import config_loader
from src.dataset_labels import DatasetLabelProvider, LogEntryMissingLabelError

_LOGGER = logging.getLogger(__name__)


class OutputFileNames(object):
    RESULTS_LIST = "raw_performace_results.yml"
    ROC_CURVE_DATA = "roc_curve_data.yml"
    ROC_CURVE_PLOT = "roc_curve_plot.pdf"
    PRECISION_RECALL_FSCORE_PLOT = "precision_recall_fscore_plot.pdf"
    ANOMALY_DATA_PLOT = "anomaly_data_plot.pdf"
    ROC_CURVE_SKLEARN_PLOT = "roc_curve_sklearn_plot.pdf"


class AnomalyAnalysisLabeledConfig(object):
    def __init__(self, config_dict):
        self._raw_dataset_path = config_dict["raw_dataset_path"]
        self._anomaly_values_path = config_dict["anomaly_values_path"]
        self._labels_file_path = config_dict["labels_file_path"]
        self._tf_vector_file_path = config_dict["tf_labels_file_path"]
        self._threshold_num = config_dict["threshold_num"]
        self._output_folder = config_dict["output_folder"]

    @property
    def raw_dataset_path(self):
        return self._raw_dataset_path

    @property
    def anomaly_values_path(self):
        return self._anomaly_values_path

    @property
    def labels_file_path(self):
        return self._labels_file_path

    @property
    def tf_vector_file_path(self):
        return self._tf_vector_file_path

    @property
    def threshold_num(self):
        return self._threshold_num

    @property
    def output_folder(self):
        return self._output_folder


def load_anomaly_error_values(anomaly_values_path: str):
    # convert back to pickle
    with open(anomaly_values_path, "rb") as istream:
        return pickle.load(istream)


def extract_labels_for_dataset(dataset_label_provider: DatasetLabelProvider, raw_dataset_path):
    anomalies = list()
    with open(raw_dataset_path, "r") as istream:
        for line in istream:
            try:
                anomalies.append(
                    dataset_label_provider.is_log_entry_anomalous(line)
                )
            except LogEntryMissingLabelError:
                anomalies.append(None)
                _LOGGER.info("Missing label for line {}".format(line))
            except KeyError as err:
                anomalies.append(None)
                _LOGGER.error("No key for line {}. Error {}".format(line, err))

    return anomalies


def _get_threshold_set(error_values, threshold_number):
    med, min_val = np.median(error_values), min(error_values)
    lb, ub = med, med + (med - min_val) * 5
    return np.linspace(
        start=lb, stop=ub, num=threshold_number
    )


def _get_accuracy_values(error_values, labels, threshold):
    is_anomaly_by_threshold = [
        err_value > threshold for err_value in error_values
    ]

    true_pos, true_neg, false_positives_cnt, false_negatives_cnt = (
        0, 0, 0, 0
    )
    for is_anomaly_thresh, is_anomaly_label in zip(is_anomaly_by_threshold, labels):
        if is_anomaly_label is None:
            continue
        if is_anomaly_thresh and is_anomaly_label:
            true_pos += 1
        elif not is_anomaly_thresh and not is_anomaly_label:
            true_neg += 1
        elif is_anomaly_label and not is_anomaly_thresh:
            false_negatives_cnt += 1
        elif not is_anomaly_label and is_anomaly_thresh:
            false_positives_cnt += 1
    ###
    # The following has to be done because it can't work with None values
    new_a, new_b = list(), list()
    for is_anomaly_thresh, is_anomaly_label in zip(is_anomaly_by_threshold, labels):
        if is_anomaly_label is None:
            continue
        new_a.append(is_anomaly_label)
        new_b.append(is_anomaly_thresh)
    prec, recall, fbeta, supp = precision_recall_fscore_support(new_a, new_b)
    ###

    dataset_len = len(error_values)
    return {
        "threshold": threshold,

        "true_positives": true_pos,
        "true_negatives": true_neg,
        "false_positives": false_positives_cnt,
        "false_negatives": false_negatives_cnt,
        "dataset_length": dataset_len,

        "positive_predictive_value": true_pos / (true_pos + false_positives_cnt),
        "negative_predictive_value": true_neg / (true_neg + false_negatives_cnt),
        "false_positives_rate": false_positives_cnt / (false_positives_cnt + true_neg),
        "false_negatives_rate": false_negatives_cnt / (false_negatives_cnt + true_pos),

        "precision": prec.tolist(),
        "recall": recall.tolist(),
        "fbeta": fbeta.tolist(),
        "support": supp.tolist()
    }


def get_roc_curve(results_list):
    thresholds = [
        results["threshold"] for results in results_list
    ]

    true_positives_rate = [
        results["positive_predictive_value"] for results in results_list
    ]

    false_positives_rate = [
        results["false_positives_rate"] for results in results_list
    ]

    return [
        (th, tpr, fpr) for th, tpr, fpr in zip(thresholds, true_positives_rate, false_positives_rate)
    ]


def plot_roc_curve(roc_curve, output_path):
    tpr_points = [
        roc_val[1] for roc_val in roc_curve
    ]

    fpr_points = [
        roc_val[2] for roc_val in roc_curve
    ]
    plt.plot(fpr_points, tpr_points, "r")

    # plot 45 degree line
    r_lim = max(fpr_points)
    x = np.linspace(0, r_lim)
    plt.plot(x, x)

    plt.savefig(output_path)
    plt.clf()


def plot_sklearn_roc_curve(error_values, true_labels, output_path):
    ###
    # The following has to be done because it can't work with None values
    new_err_vals, new_true_labels = list(), list()
    for err_val, true_label in zip(error_values, true_labels):
        if true_label is None:
            continue
        new_err_vals.append(err_val)
        new_true_labels.append(true_label)
    ###

    fpr, tpr, thresholds = roc_curve(new_true_labels, new_err_vals)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig(output_path)
    plt.clf()


def plot_precision_recall_fscore(results_list, output_path):
    thresholds = [
        results["threshold"] for results in results_list
    ]

    precision = [
        results["precision"][1] for results in results_list
    ]

    recall = [
        results["recall"][1] for results in results_list
    ]

    fscore = [
        results["fbeta"][1] for results in results_list
    ]

    plt.plot(thresholds, precision, "r")
    plt.plot(thresholds, recall, "g")
    plt.plot(thresholds, fscore, "b")

    plt.savefig(output_path)
    plt.clf()


def plot_anomaly_data(error_values, output_path, threshold=None, limit_x=None):
    if limit_x is not None:
        plt.plot(error_values[:limit_x])
    else:
        plt.plot(error_values)

    if threshold is not None and limit_x is not None:
        plt.hlines(threshold, 0, limit_x)
    elif threshold is not None and limit_x is None:
        plt.hlines(threshold, 0, len(error_values) - 1, "r")

    plt.savefig(output_path)
    plt.clf()


def get_results(error_values, labels, threshold_number):
    thresholds = _get_threshold_set(error_values, threshold_number)

    results_list = list()
    for thresh in thresholds:
        results_list.append(
            _get_accuracy_values(
                error_values=error_values,
                labels=labels,
                threshold=int(thresh)
            )
        )

    return results_list


def anomaly_analysis_labeled_main(config: AnomalyAnalysisLabeledConfig):
    _LOGGER.info("Getting dataset label provider")
    dataset_label_provider = DatasetLabelProvider(
        labels_file_path=config.labels_file_path,
        tf_vector_file_path=config.tf_vector_file_path
    )
    _LOGGER.info("Extracting dataset labels")
    dataset_labels = extract_labels_for_dataset(
        dataset_label_provider, config.raw_dataset_path
    )

    _LOGGER.info("Loading anomaly error values")
    error_values = load_anomaly_error_values(config.anomaly_values_path)

    _LOGGER.info("Getting results")
    results_list = get_results(
        error_values=error_values,
        labels=dataset_labels,
        threshold_number=config.threshold_num
    )

    results_list_output_path = os.path.join(config.output_folder, OutputFileNames.RESULTS_LIST)
    _LOGGER.info("Saving results list to {}".format(results_list_output_path))
    with open(results_list_output_path, "w") as ostream:
        yaml.dump(results_list, ostream)

    roc_curve_data_output_path = os.path.join(config.output_folder, OutputFileNames.ROC_CURVE_DATA)
    _LOGGER.info("Saving ROC curve data to {}".format(roc_curve_data_output_path))
    roc_curve_data = get_roc_curve(results_list)
    with open(roc_curve_data_output_path, "w") as ostream:
        yaml.dump(roc_curve_data, ostream)

    roc_curve_plot_output_path = os.path.join(config.output_folder, OutputFileNames.ROC_CURVE_PLOT)
    _LOGGER.info("Saving ROC curve plot to {}".format(roc_curve_plot_output_path))
    plot_roc_curve(roc_curve_data, roc_curve_plot_output_path)

    anomaly_data_plot_output_path = os.path.join(config.output_folder, OutputFileNames.ANOMALY_DATA_PLOT)
    _LOGGER.info("Saving anomaly data plot to {}".format(anomaly_data_plot_output_path))
    plot_anomaly_data(error_values, anomaly_data_plot_output_path)

    precision_recall_fscore_plot_output_path = os.path.join(
        config.output_folder, OutputFileNames.PRECISION_RECALL_FSCORE_PLOT
    )
    _LOGGER.info("Saving precision-recall-fscore plot to {}".format(precision_recall_fscore_plot_output_path))
    plot_precision_recall_fscore(results_list, precision_recall_fscore_plot_output_path)

    roc_curve_sklearn_plot_output_path = os.path.join(config.output_folder, OutputFileNames.ROC_CURVE_SKLEARN_PLOT)
    _LOGGER.info("Plotting ROC curve according to sk-learn to {}".format(roc_curve_sklearn_plot_output_path))
    plot_sklearn_roc_curve(error_values, dataset_labels, roc_curve_sklearn_plot_output_path)

    _LOGGER.info("Finished.")


if __name__ == '__main__':
    configs = config_loader.initialize_program()
    anomaly_analysis_labeled_main(configs.anomaly_detector_eval_labeled)
