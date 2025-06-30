from fair_test.datasets.custom_dataset import Custom_Dataset

class Mnist(Custom_Dataset):
    def __init__(self):
        super().__init__(hf_id="mnist", y_label="label", n_classes=10, n_features=28 * 28)

    def get_sensitive_features(self):
        # MNIST does not have a sensitive feature, return an empty string
        return ["image"]
    def get_n_classes(self):
        return self.n_classes
    def get_n_features(self):
        return self.n_features