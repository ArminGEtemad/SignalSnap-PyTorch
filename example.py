from MultiSS_SpectrumConfig import SpectrumConfig, DataImportConfig
from MultiSS_SpectrumCalculaotr import SpectrumCalculator
from MultiSS_CrossConfig import CrossConfig
from MultiSS_PlotConfig import PlotConfig
from MultiSS_SpectrumPlotter import SpectrumPlotter

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# testing
N = int(1e6)
data1 = np.sin(np.linspace(0, 50000*np.pi, N)) + 3
data2 = np.cos(np.linspace(0, 50000*np.pi, N)) + 3
data3 = np.random.rand(N)
#data4 = np.cos(np.linspace(0, 50000*np.pi, N)) + 9

config1 = DataImportConfig(data=data1)
config2 = DataImportConfig(data=data2)
config3 = DataImportConfig(data=data3)
#config4 = DataImportConfig(data=data4)

sconfig = SpectrumConfig(dt=1, f_unit='Hz', backend='cpu', order_in=[2], spectrum_size=1000, show_first_frame=False)
selected_data = [0, 1]
cconfig = CrossConfig(auto_corr=True, cross_corr_2=None)
scalc = SpectrumCalculator(sconfig, cconfig, [config1, config2, config3], selected=selected_data)

pconfig = PlotConfig(f_min=0, f_max=0.2, display_orders=None, significance=1, arcsinh_scale=(True, 0.02), plot_format=None)

scalc.calc_spec()

plotter = SpectrumPlotter(sconfig, cconfig, scalc, pconfig)
plotter.display()
#print(scalc.s)
#print('----------------------------')
#print(calc.s_err)
#print('----------------------------')
#calc.display()
#plt.plot(calc.freq[0][2], calc.s[0][2].real)
#plt.plot(calc.freq[1][2], calc.s[1][2].real)
#plt.plot(calc.freq[(1, 0)][2], calc.s[(1, 0)][2].imag)
#plt.plot(calc.freq[(0, 1)][2], calc.s[(0, 1)][2].imag)
#plt.plot(calc.freq[(0, 2)][2], calc.s[(0, 2)][2].imag)
#plt.show()
#print(calc.freq)

# ---- for the hdf5 files ---
# Configuration for importing multiple datasets from HDF5
#config1 = DataImportConfig(path="data.h5", group_key="group1", dataset="dataset1")
#config2 = DataImportConfig(path="data.h5", group_key="group1", dataset="dataset2")
#config3 = DataImportConfig(path="data.h5", group_key="group2", dataset="dataset3")

# Load the data
#config1.load_hdf5()
#config2.load_hdf5()
#config3.load_hdf5()

# Access the loaded data
#data1 = config1.data
#data2 = config2.data
#data3 = config3.data