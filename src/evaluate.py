import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch.nn as nn
from model import build_mlp
from load import build_dataloaders
from load import load_numpy_data
from load import load_str_labels
from sklearn.preprocessing import RobustScaler
import sklearn.metrics as metrics
import files as f
import argparse
import os
import math
import sys
import progressbar
import torch
import joblib

def initialize_nn(model_file):
    '''Initializes and loads a saved neural network model.'''
    dataloaders, attrib_dict = build_dataloaders()
    device = torch.device('cpu')
    model = build_mlp(dataloaders, attrib_dict, device)
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
    except Exception as e:
        print("Could not load MLP. " + str(e))
        sys.exit()
    model.eval()
    test_loader = dataloaders['test']
    test_attrib = attrib_dict['test']
    return test_attrib, test_loader, model


def initialize_sklearn_model(file_name, should_reduce=False):
    '''Initializes and loads a saved sklearn model. SVM or CART for our purposes.'''
    test_inputs, test_outputs, test_attrib = load_numpy_data(f.TEST)
    try:
        model = joblib.load(file_name)
    except Exception as e:
        print("Could not load model. " + str(e))
        sys.exit()
    scaler = RobustScaler()
    test_inputs = scaler.fit(test_inputs).transform(test_inputs)
    if should_reduce:
        try:
            svd = joblib.load(f.SVD)
        except Exception as e:
            print("Could not load feature selector. " + str(e))
            sys.exit()
        test_inputs = svd.transform(test_inputs)
    return test_attrib, test_inputs, test_outputs, model


def confuse_sklearn(test_attrib, test_outputs, new_outputs):
    '''Builds a confusion matrix for an sklearn model.'''
    n_categories = test_attrib.num_classes
    all_categories = test_attrib.str_outputs
    confusion = torch.zeros(n_categories, n_categories)
    confusion = update_confusion_matrix(confusion, new_outputs, test_outputs,
                                        len(test_outputs), category_index_from_output_np)
    normalize_and_plot(all_categories, n_categories, confusion)


def evaluate_nn(dataloader, model, n_categories, n_examples, novel):
    '''Runs a neural network on the test data and stores the performance in a
    confusion matrix.'''
    confusion = torch.zeros(n_categories, n_categories)
    batch_size = dataloader.batch_size
    num_batches = math.ceil(float(n_examples) / batch_size)

    if novel:
        all_outputs = torch.zeros(n_examples)
        all_labels = torch.zeros(n_examples)

    with progressbar.ProgressBar(max_value=num_batches, redirect_stdout=True) as bar:
        for step, (inputs, outputs) in enumerate(dataloader):
            new_output = model(inputs)
            confusion = update_confusion_matrix(
                confusion, new_output, outputs, outputs.shape[0], category_index_from_output_tensor)
            if novel:
                _, preds = torch.max(new_output, 1)
                all_outputs[step * batch_size:step * batch_size + outputs.shape[0]] = preds
                all_labels[step * batch_size:step * batch_size + outputs.shape[0]] = outputs

            bar.update(step)
    if novel:
        evaluate_novel(all_outputs.numpy(), all_labels.numpy())

    return confusion


def evaluate_combined(dataloader, tree_model, svm_model, mlp_model, n_categories, n_examples, combo_vote, novel):
    '''Runs all 3 models combined on the test data and stores the performance in a
    confusion matrix.'''
    confusion = torch.zeros(n_categories, n_categories)
    batch_size = dataloader.batch_size
    num_batches = math.ceil(float(n_examples) / batch_size)
    scaler = RobustScaler()
    svd = joblib.load(f.SVD)

    if novel:
        all_outputs = torch.zeros(n_examples)
        all_labels = torch.zeros(n_examples)

    with progressbar.ProgressBar(max_value=num_batches, redirect_stdout=True) as bar:
        for step, (inputs, outputs) in enumerate(dataloader):
            sk_inputs = inputs.numpy()
            sk_inputs = scaler.fit(sk_inputs).transform(sk_inputs)
            tree_new_output = tree_model.predict(sk_inputs)
            svm_inputs = svd.transform(sk_inputs)
            svm_new_output = svm_model.predict(svm_inputs)
            mlp_new_output = mlp_model(inputs)
            new_output = torch.zeros(inputs.shape[0], 2)
            for i in range(inputs.shape[0]):
                c1 = tree_new_output[i]
                c2 = svm_new_output[i]
                c3 = category_index_from_output_tensor(mlp_new_output[i])
                # detect threat based on vote
                # combined-1: c1 + c3 > 0
                # combined-2: c1 + c2 + c3 > 2
                if novel:
                    all_labels[step * batch_size + i] = outputs[i]
                if combo_vote(c1, c2, c3):
                    new_output[i][1] += 1
                    if novel:
                        all_outputs[step * batch_size + i] = 1
                else:
                    new_output[i][0] += 1
                    if novel:
                        all_outputs[step * batch_size + i] = 0

            confusion = update_confusion_matrix(
                confusion, new_output, outputs, outputs.shape[0], category_index_from_output_tensor)
            bar.update(step)

    if novel:
        evaluate_novel(all_outputs.numpy(), all_labels.numpy())

    return confusion


def update_confusion_matrix(confusion, new_output, outputs, shape, indexer):
    '''Given the guessed outputs and the actual labels, returns a confusion matrix
    evaluating the accuracy of the model.'''
    for i in range(shape):
        guess_i = indexer(new_output[i])
        category_i = outputs[i].item()
        confusion[category_i][guess_i] += 1
    return confusion


def category_index_from_output_tensor(output):
    '''Find the max value in the tensor and return its index, determining the category.
    '''
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i


def category_index_from_output_np(output):
    return output


def confuse_nn(test_attrib, test_loader, model, novel):
    '''Evaluates a neural network model and graphs a confusion matrix.'''
    n_categories = test_attrib.num_classes
    all_categories = test_attrib.str_outputs
    n_examples = test_attrib.num_examples
    confusion = evaluate_nn(test_loader, model, n_categories, n_examples, novel)
    normalize_and_plot(all_categories, n_categories, confusion)


def normalize_and_plot(all_categories, n_categories, confusion):
    '''Normalizes a confusion matrix, prints it, and generates a graphical plot.'''
    print("Plotting")
    acc = ((confusion[1][1] + confusion[0][0]) / confusion.sum()).item()

    precision = float(confusion[1][1].item()) / (confusion[1][1].item() + confusion[0][1].item())

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    true_pos = confusion[1][1].item()
    true_neg = confusion[0][0].item()
    false_pos = confusion[0][1].item()
    false_neg = confusion[1][0].item()
    f1 = 2 * (precision * true_pos) / (precision + true_pos)

    print("True positive rate: " + str(true_pos))
    print("True negative rate: " + str(true_neg))
    print("False positive rate: " + str(false_pos))
    print("False negative rate: " + str(false_neg))
    print("Accuracy: " + str(acc))
    print("Precision: " + str(precision))
    print("F1 Score: " + str(f1))


    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xlabel("predicted")
    ax.xaxis.set_label_position('top')
    plt.ylabel("actual")

    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluate_sklearn(file_name, novel, should_reduce=False):
    '''Initializes and evaluates an sklearn model and graphs the results.'''
    test_attrib, test_inputs, test_outputs, model = initialize_sklearn_model(file_name,
                                                                          should_reduce)
    new_outputs = model.predict(test_inputs)
    if novel:
        evaluate_novel(new_outputs, test_outputs)
    confuse_sklearn(test_attrib, test_outputs, new_outputs)

def evaluate_novel(outputs, labels):
    '''Tests how many of the shared and novel attack types the model got correct.'''
    str_labels, all_test_labels = load_str_labels()
    for i in range(outputs.shape[0]):
        if outputs[i] == labels[i]:
            for t in ['novel', 'shared']:
                if all_test_labels[i] in str_labels[t].keys():
                    str_labels[t][all_test_labels[i]]['correct'] += 1

    for t in ['novel', 'shared']:
        print('===================================')
        print(t + ' record types classified correctly')
        print('===================================')
        total_total = 0
        total_correct = 0
        for s in sorted(str_labels[t]):
            correct = str_labels[t][s]['correct']
            total = str_labels[t][s]['total']
            acc = 0
            if total > 0:
                acc = float(correct) / total
            print('{}: {:0.3f}, {} out of {}'.format(s.decode('utf-8'), acc, correct, total))
            total_total += total
            total_correct += correct
        total_acc = 0
        if total_total > 0:
            total_acc = float(total_correct) / total_total
        print('total: {:0.3f}, {} out of {}'.format(total_acc, total_correct, total_total))


def evaluate_roc_nn(dataloader, model, n_examples):
    '''Runs a neural network on the test data and graphs the ROC curve.'''
    batch_size = dataloader.batch_size
    num_batches = math.ceil(float(n_examples) / batch_size)
    soft_max = nn.Softmax(dim=1)
    prob_outputs = torch.zeros(n_examples)
    test_outputs = torch.zeros(n_examples)
    with progressbar.ProgressBar(max_value=num_batches, redirect_stdout=True) as bar:
        for step, (inputs, outputs) in enumerate(dataloader):
            new_output = model(inputs)
            prob_output = soft_max(new_output)
            prob_outputs[step * batch_size:step * batch_size + outputs.shape[0]] = prob_output[:, 1]
            test_outputs[step * batch_size:step * batch_size + outputs.shape[0]] = outputs

    false_pos, true_pos, _ = metrics.roc_curve(test_outputs.detach().numpy(), prob_outputs.detach().numpy())
    auc = metrics.auc(false_pos, true_pos)
    label = 'MLP on {}, AUC = {:0.3f}'.format(f.TEST, auc)
    plt.plot(false_pos, true_pos, label=label, linewidth=2.0)

def evaluate_roc_sklearn(file_name, should_reduce=False):
    '''Initializes and evaluates the probabilities of an sklearn model under different biases
    and graphs the ROC curve.'''
    test_attrib, test_inputs, test_outputs, model = initialize_sklearn_model(file_name,
                                                                          should_reduce)
    prob_outputs = model.predict_proba(test_inputs)
    false_pos, true_pos, _ = metrics.roc_curve(test_outputs, prob_outputs[:, 1])
    auc = metrics.auc(false_pos, true_pos)
    label = ''
    if should_reduce:
        label += 'SVM on '
    else:
        label += 'CART on '
    label += f.TEST
    label += ', AUC = %0.3f' % auc
    plt.plot(false_pos, true_pos, label=label, linewidth=2.0)

def graph_roc(zoom):
    '''Shows the ROC curve graph. Adapted from
    https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python.'''
    title = 'Receiver Operating Characteristic'
    if zoom < 1:
        title += ' (zoomed into top left)'
    plt.title(title)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--', linewidth=2.0)
    plt.xlim([0, zoom])
    plt.ylim([1-zoom, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def initialize_and_evaluate_combined(combo_vote, novel):
    '''Initializes and evaluates a combined model.'''
    test_attrib, test_inputs, test_outputs, tree_model = initialize_sklearn_model(
        f.TREE)
    test_attrib, test_inputs, test_outputs, svm_model = initialize_sklearn_model(
        f.SVM)
    test_attrib, test_loader, mlp_model = initialize_nn(f.MLP)
    n_categories = test_attrib.num_classes
    all_categories = test_attrib.str_outputs
    n_examples = test_attrib.num_examples
    confusion = evaluate_combined(
        test_loader, tree_model, svm_model, mlp_model, n_categories, n_examples, combo_vote, novel)
    normalize_and_plot(all_categories, n_categories, confusion)

def combo_vote_1(c1, c2, c3):
    return c1 + c3 > 0

def combo_vote_2(c1, c2, c3):
    return c1 + c2 + c3 > 2

def evaluate_roc_all():
    '''Constructs the three models' ROC curves.'''
    evaluate_roc_sklearn(f.SVM, should_reduce=True)
    evaluate_roc_sklearn(f.TREE)
    test_attrib, test_loader, model = initialize_nn(f.MLP)
    evaluate_roc_nn(test_loader, model, test_attrib.num_examples)

def limit_range(arg):
    '''Limit a command line argument between 0 and 1. Adapted from
    https://mail.python.org/pipermail/tutor/2013-January/093635.html
    '''
    try:
        value = float(arg)
    except ValueError as err:
       raise argparse.ArgumentTypeError(str(err))

    if value < 1e-8 or value > 1:
        message = "Expected 0 < value < 1, got value = {}".format(value)
        raise argparse.ArgumentTypeError(message)

    return value

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate and graph the perfomance of ML models. Default=multilayer perceptron.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--svm",
                       help="Evaluate the support vector machine.",
                       action="store_true")
    group.add_argument("--mlp",
                       help="Evaluate the multilayer perceptron (neural network).",
                       action="store_true")
    group.add_argument("--tree",
                       help="Evaluate the decision tree.",
                       action="store_true")
    group.add_argument("--combined1",
                       help="Evaluate a combined model which maximizes the percentage of malicious PCAP examples we catch.",
                       action="store_true")
    group.add_argument("--combined2",
                       help="Evaluate a combined model which minimizes the false positive rate.",
                       action="store_true")
    group.add_argument("--roc",
                       help="Graph ROC curve.",
                       action="store_true")
    parser.add_argument("--zoom", type=limit_range,
                       help="Percentage of axes to zoom into from 0 to 1 for ROC curve.",
                       default=1)
    parser.add_argument("--novel",
                       help="Calculate how many novel and shared attacks were classified correctly.",
                       action="store_true")
    args = parser.parse_args()

    # Evaluate and graph
    if args.roc:
        evaluate_roc_all()
        graph_roc(args.zoom)
    elif args.svm:
        evaluate_sklearn(f.SVM, args.novel, should_reduce=True)
    elif args.tree:
        evaluate_sklearn(f.TREE, args.novel)
    elif args.combined1:
        initialize_and_evaluate_combined(combo_vote_1, args.novel)
    elif args.combined2:
        initialize_and_evaluate_combined(combo_vote_2, args.novel)
    else:
        # Default is evaluating MLP
        test_attrib, test_loader, model = initialize_nn(f.MLP)
        confuse_nn(test_attrib, test_loader, model, args.novel)
