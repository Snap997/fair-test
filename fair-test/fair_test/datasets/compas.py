from fair_test.datasets.custom_dataset import Custom_Dataset

class Compas(Custom_Dataset):
    def __init__(self):
        super().__init__(
            id="imodels/compas-recidivism",
            name="Compas Recidivism",
            n_classes=2,
            n_features=20
            )
    

    def get_sensitive_feature(self):
        return "is_recid"


