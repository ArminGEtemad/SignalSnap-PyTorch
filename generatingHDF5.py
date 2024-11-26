import numpy as np
import h5py

# Define the size of the data to be approximately 100MB
data_size_MB = 100  # Target size in MB
data_element_size = 8  # Each double (float64) element is 8 bytes
N = (data_size_MB * 1024 * 1024) // data_element_size  # Number of elements for ~100MB

# Generate the datasets
data1 = np.sin(np.linspace(0, 50000 * np.pi, int(1e6))) + 3
data2 = np.cos(np.linspace(0, 50000 * np.pi, int(1e6))) + 3

# Save the data to an HDF5 file
hdf5_file_path = "test_data.h5"
with h5py.File(hdf5_file_path, "w") as hdf5_file:
    hdf5_file.create_dataset("group1/data1", data=data1)
    hdf5_file.create_dataset("group2/data2", data=data2)

print(f"Data saved to {hdf5_file_path}")