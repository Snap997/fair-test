from fair_test.datasets.custom_dataset import Custom_Dataset

class Compas(Custom_Dataset):
    def __init__(self):
        super().__init__(
            hf_id="imodels/compas-recidivism",
            y_label="is_recid",
            n_classes=2,
            n_features=20
            )
    
    def get_sensitive_features(self):
        return ["race:African-American"]
    
    def get_y_classes(self):
        return [0, 1]  


