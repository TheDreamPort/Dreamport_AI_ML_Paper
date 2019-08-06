class DatasetAttributes:
    '''Holds information about the dataset.
    Attributes:
        str_inputs (list[str]): the strings used in the features before conversion to numbers
        str_outputs (list[str]): the strings used for the output classes
        feature_length (int): the number of features in each feature vector
        num_classes (int): the number of classes in the output
        num_examples (int): the number of examples in the dataset
    '''

    def __init__(self, str_inputs=[], str_outputs=['normal', 'malicious'],
                 feature_length=0, num_examples=0):
        self.str_inputs = str_inputs
        self.str_outputs = str_outputs
        self.feature_length = feature_length
        self.num_classes = len(str_outputs)
        self.num_examples = num_examples
