from torch.utils.data import Dataset
from src.final_package.data import corrupt_mnist

def test_my_dataset():
    """Test the datasets returned by corrupt_mnist."""
    train_set, test_set = corrupt_mnist()
    
    # Check if both datasets are instances of Dataset
    assert isinstance(train_set, Dataset), "Train set is not a Dataset"
    assert isinstance(test_set, Dataset), "Test set is not a Dataset"
    
    # Check if the train and test datasets are not empty
    assert len(train_set) > 0, "Train dataset is empty"
    assert len(test_set) > 0, "Test dataset is empty"

    # Additional checks for the structure of data
    sample_image, sample_target = train_set[0]
    assert sample_image.shape == (1, 28, 28), "Train image shape is incorrect"
    assert isinstance(sample_target.item(), int), "Train target is not an integer"

