
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr_datasets import FederatedDataset

class Custom_Dataset():
    def __init__(self, hf_id: str, y_label:str, n_classes: int, n_features: int):
        self.hf_id = hf_id
        self.y_label = y_label
        self.n_classes = n_classes  # Number of classes in the dataset
        self.n_features = n_features  # Number of features in dataset
        self.feature_keys = []
        
    def load(self, num_partitions: int, alpha: float, partition_id: int):
        if alpha == 0:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset=self.hf_id, partitioners={"train": partitioner}
            )
        else:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by=self.y_label,
                alpha=alpha,
                seed=42,
                min_partition_size=0,
            )
            fds = FederatedDataset(
                dataset=self.hf_id,
                partitioners={"train": partitioner},
            )
        dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]

        self.feature_keys = [c for c in dataset.columns if c != self.y_label]
        return dataset
    
    def get_n_classes(self):
        return self.n_classes
    
    def get_n_features(self):
        return self.n_features
    
    def get_y(self):
        return self.y_label
    
    
    