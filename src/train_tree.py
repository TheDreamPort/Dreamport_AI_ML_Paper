from load import load_numpy_data
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
import files as f

# Load the feature vectors into numpy arrays
train_inputs, train_outputs, _ = load_numpy_data(f.TRAIN)

# Train a CART model on the data
model = DecisionTreeClassifier(min_samples_leaf=128)
model.fit(train_inputs, train_outputs)
print(model)

# Save the model to disk
dump(model, f.TREE)
