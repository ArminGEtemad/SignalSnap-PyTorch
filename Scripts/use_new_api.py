from multichss.configurators import DataConfig, SpectrumConfig
from multichss.pipelines import calculate_spectra

import numpy as np


data_conf = DataConfig(data=np.random.normal(0, 1, 100000), dt=0.1)
spec_conf = SpectrumConfig(f_min=0, f_max=4, frequency_points=301, device="cuda")

result_store = calculate_spectra(spec_conf, [data_conf])
result = result_store.get((0,0,0))
if result.spectrum is not None:
    print(result.spectrum.shape)
