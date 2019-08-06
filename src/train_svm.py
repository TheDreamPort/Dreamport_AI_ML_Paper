from load import load_numpy_data
from sklearn import svm
from joblib import dump, load
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
import files as f
import argparse


def svc_param_selection(inputs, outputs, nfolds):
    '''
    Finds some reasonable SVM hyperparameters. Increasing max_iter and
    evaluating more choices of Cs and gammas will give better results but take
    much longer. Adapted from https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0.
    '''
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(
        svm.SVC(verbose=1, max_iter=100), param_grid, cv=nfolds)
    grid_search.fit(inputs, outputs)
    grid_search.best_params_
    return grid_search.best_params_


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Trains a support vector machine model on the NSL-KDD dataset.")
parser.add_argument("-C", type=float,
                    help="Low values make the model prefer simple decision functions at the cost of classification accuracy.",
                    default=10)
parser.add_argument("-g", "--gamma", type=float,
                    help="Controls how wide the influence of each example is in the problem space.",
                    default=0.01)
parser.add_argument("-i", "--maxiter", type=int,
                    help="Defines when the model stops training if it doesn't finish.",
                    default=100000)
parser.add_argument("-s", "--search",
                    help="Finds some reasonable SVM hyperparameters by performing grid search.",
                    action="store_true")
args = parser.parse_args()

# Load the feature vectors into numpy arrays
train_inputs, train_outputs, _ = load_numpy_data(f.TRAIN)

# Perform feature selection, reducing the features to 50
svd = TruncatedSVD(n_components=50)
train_inputs = svd.fit_transform(train_inputs)

if args.search:
    # Automatically detect some good values for gamma and C
    print(svc_param_selection(train_inputs, train_outputs, 5))

# Initialize the model with the hyperparameters we found
model = svm.SVC(gamma=args.gamma, C=args.C,
                verbose=1, max_iter=args.maxiter,
                probability=True)

print("Training model.")
# Train the model on the data
model.fit(train_inputs, train_outputs)

# Save the feature selector and model to disk
dump(svd, f.SVD)
dump(model, f.SVM)
