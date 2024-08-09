import h5py

def print_structure(name, obj):
    """Prints the structure of the HDF5 file."""
    print(f"Object: {name}, Type: {type(obj)}")

def print_attrs(name, obj):
    """Prints the attributes of an HDF5 object."""
    print(f"\nCalling print_attrs for {name}")
    print(f"{name}:")
    if obj.attrs:
        for key, value in obj.attrs.items():
            print(f"  Attribute: {key} - Value: {value}")
    else:
        print("  No attributes found")

def print_datasets(name, obj):
    """Prints the datasets within the HDF5 file."""
    print(f"\nCalling print_datasets for {name}")
    if isinstance(obj, h5py.Dataset):
        print(f"\nDataset: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Data type: {obj.dtype}")

def print_root_attrs(file):
    """Prints the attributes of the root level of the HDF5 file."""
    print("\nRoot level attributes:")
    if file.attrs:
        for key, value in file.attrs.items():
            print(f"  Attribute: {key} - Value: {value}")
    else:
        print("  No attributes found at the root level")

def inspect_hdf5(file_path):
    """Inspects the HDF5 file, printing all attributes and datasets."""
    with h5py.File(file_path, 'r') as f:
        print("Inspecting HDF5 file...")
        
        # Print the structure of the file
        print("\nFile Structure:")
        f.visititems(print_structure)
        
        # Print the root level attributes
        print_root_attrs(f)
        
        # Print attributes and datasets
        f.visititems(print_attrs)
        f.visititems(print_datasets)

# Usage
file_path = '/home/jmdelahanty/OneDrive/lgfl/rawdata/sub-0001/20240718_sub-0001_OnePortEtohExperiment/20240718_sub-0001_OnePortEtohExperiment.h5'

inspect_hdf5(file_path)
