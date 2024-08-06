import h5py
def print_attrs(name, obj):
    """Prints the attributes of an HDF5 object."""
    print(f"\n{name}:")
    for key, value in obj.attrs.items():
        print(f"  Attribute: {key} - Value: {value}")

def print_datasets(name, obj):
    """Prints the datasets within the HDF5 file."""
    if isinstance(obj, h5py.Dataset):
        print(f"\nDataset: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Data type: {obj.dtype}")

def inspect_hdf5(file_path):
    """Inspects the HDF5 file, printing all attributes and datasets."""
    with h5py.File(file_path, 'r') as f:
        print("Inspecting HDF5 file...")
        f.visititems(print_attrs)
        f.visititems(print_datasets)

# Usage
file_path = '/home/jmdelahanty/OneDrive/lgfl/rawdata/sub-0001/20240718_sub-0001_OnePortEtohExperiment/20240718_sub-0001_OnePortEtohExperiment.h5'
inspect_hdf5(file_path)