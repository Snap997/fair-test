from fair_test.datasets.custom_dataset import Custom_Dataset

class Cifar(Custom_Dataset):
    def __init__(self):
        super().__init__(id="cifar10", name="CIFAR-10", n_classes=10, n_features=32 * 32 * 3)

    def get_sensitive_feature(self):
        # CIFAR-10 does not have a sensitive feature, return an empty string
        return "label"
    
    def get_n_classes(self):
        return self.n_classes
    
    def get_n_features(self):
        return self.n_features