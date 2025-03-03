import numpy as np
import h5py

class SpectrumConfig:
    def __init__(self, dt, f_unit='Hz', f_max=None, f_min=0, f_lists=None, full_bispectrum=False,
                 backend='mps', spectrum_size=100, order_in='all',
                 coherent=False, m=10, m_var=10, show_first_frame=True, break_after=int(1e6)):
        self.dt = dt
        self.f_unit = f_unit
        self.f_max = f_max
        self.f_min = f_min
        self.f_lists = f_lists
        self.backend = backend
        self.spectrum_size = spectrum_size
        self.order_in = order_in
        self.coherent = coherent
        self.m = m
        self.m_var = m_var
        self.show_first_frame = show_first_frame
        self.break_after = break_after

class DataImportConfig:
    def __init__(self, data=None, path=None, group_key=None, dataset=None, dt=None):
        self.data = data
        self.path = path
        self.group_key = group_key
        self.dataset = dataset
        self.dt = dt

    @staticmethod
    def data_config_dic(data_config_list):
        return {config.data: config for config in data_config_list}
