import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm.auto import tqdm

import torch
from numba import njit
from scipy.fft import rfftfreq
from MultiSS_SpectrumConfig import SpectrumConfig, DataImportConfig
from MultiSS_CrossConfig import CrossConfig
import pandas as pd
from tabulate import tabulate

def load_spec(path):
    f = open(path, mode='rb')
    obj = pickle.load(f)
    f.close()
    return obj

def to_hdf(dt, data, path, group_name, dataset_name):
    with h5py.File(path, "w") as f:
        grp = f.create_group(group_name)
        d = grp.create_dataset(dataset_name, data=data)
        d.attrs['dt'] = dt

def unit_conversion(f_unit):
    if f_unit == 'Hz':
        t_unit = 's'
    elif f_unit == 'kHz':
        t_unit = 'ms'
    elif f_unit == 'MHz':
        t_unit = 'us'
    elif f_unit == 'GHz':
        t_unit = 'ns'
    elif f_unit == 'THz':
        t_unit = 'ps'
    else:
        raise ValueError(f'Unknown frequency unit: {f_unit}')
    return t_unit

# ---- Helper functions related to confinded Gaussian window ----
@njit
def g(x, n_windows, l, sigma_t):
    ge_e = x - n_windows/2
    ge_d = 2 * l * sigma_t

    sqrt_ge = ge_e / ge_d
    ge = - sqrt_ge*sqrt_ge
    gaus = np.exp(ge)

    return gaus

@njit
def calc_window(x, n_windows, l, sigma_t):
    term_x = g(x, n_windows, l, sigma_t)
    term_h = g(-0.5, n_windows, l, sigma_t)
    term_x_p_l = g(x + l, n_windows, l, sigma_t)
    term_x_m_l = g(x - l, n_windows, l, sigma_t)
    term_h_p_l = g(-0.5 + l, n_windows, l, sigma_t)
    term_h_m_l = g(-0.5 - l, n_windows, l, sigma_t)

    win = term_x - (term_h * (term_x_p_l + term_x_m_l)) / (term_h_p_l +
                                                           term_h_m_l)
    return win

@njit
def cg_window(n_windows, fs):
    """
    confined Gaussian window
    """
    x = np.linspace(0, n_windows, n_windows)
    l = n_windows + 1
    sigma_t = 0.14

    window = calc_window(x, n_windows, l, sigma_t)
    norm = np.sum(window*window) / fs
    window_full = window / np.sqrt(norm)

    return window_full, norm

# ---------------------------------------------------------------

class SpectrumCalculator:
    def __init__(self, sconfig: SpectrumConfig, cconfig: CrossConfig, 
                 diconfig_list: list[DataImportConfig], selected=None):
        self.sconfig = sconfig
        self.cconfig = cconfig
        self.diconfig_list = diconfig_list
        self.selected = selected if selected is not None else list(range(len(diconfig_list)))
        self.cross2_selected = (self.cconfig.cross_corr_2 
                                if hasattr(self.cconfig, 'cross_corr_2') and isinstance(self.cconfig.cross_corr_2, list)
                                else [])
        self.device = torch.device(self.sconfig.backend)
        self.t_unit = unit_conversion(sconfig.f_unit)
        self.fs = 1 / self.sconfig.dt

        # Initialize various dictionaries in one go
        self._init_dicts()
        
        self.import_data()
        self.validate_shapes()  # Crash if data shapes mismatch

        # Flag to use full FFT (for negative frequencies)
        self.use_full_fft = (self.sconfig.f_min < 0)

        # MPS backend precision support
        if self.sconfig.backend == 'mps':
            self.use_float32 = True
            print('MPS backend on Apple hardware supports single precision.\n'
                  'Using float32 for all tensors!')
        else:
            self.use_float32 = False

    def _init_dicts(self):
        # Helper to initialize dictionaries for both selected and cross datasets
        self.n_chunks = {i: 0 for i in self.selected}
        self.n_chunks.update({(k1, k2): 0 for k1, k2 in self.cross2_selected})

        self.m = {1: None, 2: None, 4: None}
        self.m_var = {1: None, 2: None, 4: None}

        # For frequency, spectra, error, etc.
        keys = self.selected + self.cross2_selected
        self.freq = {key: {2: None, 4: None} for key in keys}
        self.f_lists = {key: {2: None, 3: None, 4: None} for key in keys}

        self.s = {key: {1: None, 2: None, 4: None} for key in self.selected}
        self.s.update({key: {2: None, 4: None} for key in self.cross2_selected})
        self.s_gpu = {key: {1: None, 2: None, 4: None} for key in self.selected}
        self.s_gpu.update({key: {2: None, 4: None} for key in self.cross2_selected})
        self.s_err = {key: {1: None, 2: None, 4: None} for key in self.selected}
        self.s_err.update({key: {2: None, 4: None} for key in self.cross2_selected})
        self.s_err_gpu = {key: {1: None, 2: None, 4: None} for key in self.selected}
        self.s_err_gpu.update({key: {2: None, 4: None} for key in self.cross2_selected})
        self.s_errs = {key: {1: None, 2: [], 4: []} for key in self.selected}
        self.s_errs.update({key: {2: [], 4: []} for key in self.cross2_selected})
        self.err_counter = {key: {1: 0, 2: 0, 4: 0} for key in self.selected}
        self.err_counter.update({key: {2: 0, 4: 0} for key in self.cross2_selected})
        self.n_error_estimates = {key: {1: 0, 2: 0, 4: 0} for key in self.selected}
        self.n_error_estimates.update({key: {2: 0, 4: 0} for key in self.cross2_selected})

    def validate_shapes(self):
        expected_shape = self.diconfig_list[self.selected[0]].data.shape[0]
        for data_config in self.diconfig_list:
            if data_config.data.shape[0] != expected_shape:
                raise ValueError('Imported data must have same length!')

    def import_data(self):
        for data_config in self.diconfig_list:
            if data_config.data is None and data_config.path is not None:
                with h5py.File(data_config.path, 'r') as main:
                    if not data_config.group_key:
                        main_data = main[data_config.dataset]
                    else:
                        main_data = main[data_config.group_key][data_config.dataset]
                    if data_config.dt is None:
                        data_config.dt = main_data.attrs.get('dt', None)
                    data_config.data = main_data[()]
                    print(f"Data loaded from {data_config.path}")

    def plot_first_frames(self, selected, window_size):
        n_plots = len(selected)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
        if n_plots == 1:
            axes = [axes]
        for i, idx in enumerate(selected):
            data_config = self.diconfig_list[idx]
            first_frame = data_config.data[:window_size]
            t = np.arange(len(first_frame)) * self.sconfig.dt
            axes[i].plot(t, first_frame)
            axes[i].set_xlim([0, t[-1]])
            axes[i].set_title(f'First frame for data {idx}')
            axes[i].set_xlabel(f't / ({self.t_unit})')
            axes[i].set_ylabel('Amplitude')
        plt.tight_layout()
        plt.show()

    # ---- Calculating unbiased cumulants ----
    def c1(self, a_w):
        s1 = torch.mean(a_w, dim=0)
        if self.use_full_fft:
            dc_index = s1.shape[0] // 2
            return s1[dc_index]
        else:
            return s1[0]

    def c2(self, a_w1, a_w2):
        a_w2_star = torch.conj(a_w2)
        term_1 = torch.mean(a_w1 * a_w2_star, dim=0)
        factor = self.sconfig.m / (self.sconfig.m - 1)
        term_2 = torch.mean(a_w1, dim=0) * torch.mean(a_w2_star, dim=0)
        s2 = factor * (term_1 - term_2)
        return s2.squeeze(-1)


    """
    def c4(self, a_w1, a_w2):

        m = self.sconfig.m

        x = a_w1
        z = a_w2

        y = torch.conj(x)
        w = torch.conj(z)

        x_mean = x - torch.mean(x, dim=0, keepdim=True)#.repeat(1, 1, x.shape[2]) # is repeat necessary or not?
        y_mean = y - torch.mean(x, dim=0, keepdim=True)#.repeat(1, 1, y.shape[2])
        z_mean = z - torch.mean(x, dim=0, keepdim=True)#.repeat(1, 1, z.shape[2])
        w_mean = w - torch.mean(x, dim=0, keepdim=True)#.repeat(1, 1, w.shape[2])

        xyzw = torch.matmul(x_mean * y_mean, (z_mean * w_mean).transpose(-1, -2) )
        xyzw_mean = torch.mean(xyzw, dim=0)

        xy_mean = torch.mean(x_mean * y_mean, dim=0)
        zw_mean = torch.mean(z_mean * w_mean, dim=0)
        xy_zw_mean = torch.matmul(xy_mean, zw_mean.transpose(-1, -2) )

        xz_mean = torch.mean(torch.matmul(x_mean, z_mean.transpose(-1, -2) ), dim=0)
        yw_mean = torch.mean(torch.matmul(y_mean, w_mean.transpose(-1, -2) ), dim=0)
        xz_yw_mean = xz_mean * yw_mean

        xw_mean = torch.mean(torch.matmul(x_mean, w_mean.transpose(-1, -2) ), dim=0)
        yz_mean = torch.mean(torch.matmul(y_mean, z_mean.transpose(-1, -2) ), dim=0)
        xw_yz_mean = xw_mean * yz_mean


        s4 = m ** 2 / ((m - 1) * (m - 2) * (m - 3)) * (
                (m + 1) * xyzw_mean -
                (m - 1) * (
                        xy_zw_mean + xz_yw_mean + xw_yz_mean
                )
        )

        return s4
    """
    def c4(self, a_w1, a_w2, a_w3, a_w4):

        m = self.sconfig.m

        x = a_w1
        y = torch.conj(a_w2)
        z = a_w3
        w = torch.conj(a_w4)


        x_mean = x - x.mean(dim=0, keepdim=True)
        y_mean = y - y.mean(dim=0, keepdim=True)
        z_mean = z - z.mean(dim=0, keepdim=True)
        w_mean = w - w.mean(dim=0, keepdim=True)

        # Equivalent to af.matmulNT(x_mean * y_mean, z_mean * w_mean)
        xyzw = torch.matmul((x_mean * y_mean), (z_mean * w_mean).transpose(-1, -2))
        xyzw_mean = xyzw.mean(dim=0)

        # Compute all the partial means
        xy_mean = (x_mean * y_mean).mean(dim=0)
        zw_mean = (z_mean * w_mean).mean(dim=0)
        xy_zw_mean = torch.matmul(xy_mean, zw_mean.transpose(-1, -2))

        xz_mean = torch.matmul(x_mean, z_mean.transpose(-1, -2)).mean(dim=0)
        yw_mean = torch.matmul(y_mean, w_mean.transpose(-1, -2)).mean(dim=0)
        xz_yw_mean = xz_mean * yw_mean

        xw_mean = torch.matmul(x_mean, w_mean.transpose(-1, -2)).mean(dim=0)
        yz_mean = torch.matmul(y_mean, z_mean.transpose(-1, -2)).mean(dim=0)
        xw_yz_mean = xw_mean * yz_mean

        # Final combination

        s4 = m**2 / ((m - 1)*(m - 2)*(m - 3)) * (
                (m + 1)*xyzw_mean - (m - 1)*(xy_zw_mean + xz_yw_mean + xw_yz_mean)
            )

        return s4

    # ----------------------------------------

    def store_sum_single_spectrum(self, single_spectrum, order, dataset_idx):
        if self.s_gpu[dataset_idx][order] is None:
            self.s_gpu[dataset_idx][order] = single_spectrum
        else:
            self.s_gpu[dataset_idx][order] += single_spectrum

        if order == 1:
            self.s_errs[dataset_idx][order][0, self.err_counter[dataset_idx][order]] = single_spectrum
        elif order == 2:
            self.s_errs[dataset_idx][order][:, self.err_counter[dataset_idx][order]] = single_spectrum
        else:
            self.s_errs[dataset_idx][order][:, :, self.err_counter[dataset_idx][order]] = single_spectrum

        self.err_counter[dataset_idx][order] += 1

        if self.err_counter[dataset_idx][order] % self.sconfig.m_var == 0:
            dim = 1 if order in [1, 2] else 2
            self.n_error_estimates[dataset_idx][order] += 1
            factor = self.sconfig.m_var / (self.sconfig.m_var - 1)
            mean_squared = torch.mean(self.s_errs[dataset_idx][order] ** 2, dim=dim)
            squared_mean = torch.mean(self.s_errs[dataset_idx][order], dim=dim) ** 2
            s_err_gpu = factor * (mean_squared - squared_mean) / self.sconfig.m_var
            if self.s_err[dataset_idx][order] is None:
                self.s_err[dataset_idx][order] = s_err_gpu.cpu().numpy()
            else:
                self.s_err[dataset_idx][order] += s_err_gpu.cpu().numpy()
            self.err_counter[dataset_idx][order] = 0

    def store_final_spectrum(self, orders, n_chunks, dataset_idx):
        for order in orders:
            if self.s_gpu.get(dataset_idx, {}).get(order) is not None:
                self.s_gpu[dataset_idx][order] /= n_chunks
                self.s[dataset_idx][order] = self.s_gpu[dataset_idx][order].cpu().resolve_conj().numpy()
                self.s_err[dataset_idx][order] = (
                    (1 / self.n_error_estimates[dataset_idx][order]) * np.sqrt(self.s_err[dataset_idx][order])
                ) / 2  # for interlaced calculation

    def fourier_coeffs_to_spectra(self, orders, coeffs_gpu, f_min_idx, f_max_idx, single_window, dataset_idx):
        for order in orders:
            if order == 1:
                a_w = coeffs_gpu[:, f_min_idx:f_max_idx, :]
                single_spectrum = self.c1(a_w) / (self.sconfig.dt * single_window.mean() * single_window.shape[0])
            elif order == 2:
                a_w = coeffs_gpu[:, f_min_idx:f_max_idx, :] if self.sconfig.f_lists is None else coeffs_gpu
                single_spectrum = self.c2(a_w, a_w) / (self.sconfig.dt * (single_window ** 2).sum())

            elif order == 4:
                a_w = coeffs_gpu[:, f_min_idx:f_max_idx, :]
                single_spectrum = self.c4(a_w, a_w, a_w, a_w) / (self.sconfig.dt * (single_window ** 4).sum())

            self.store_sum_single_spectrum(torch.conj(single_spectrum), order, dataset_idx)


    def fourier_coeffs_to_cross_spectra(self, orders, coeffs_gpu_dict, f_min_idx, f_max_idx, single_window, key1, key2):
        for order in orders:
            if order == 2:
                if self.sconfig.f_lists is None:
                    a_w1 = coeffs_gpu_dict[key1][:, f_min_idx:f_max_idx, :]
                    a_w2 = coeffs_gpu_dict[key2][:, f_min_idx:f_max_idx, :]
                else:
                    a_w1 = coeffs_gpu_dict[key1]
                    a_w2 = coeffs_gpu_dict[key2]
                single_spectrum = self.c2(a_w1, a_w2) / (self.sconfig.dt * (single_window ** 2).sum())

            self.store_sum_single_spectrum(torch.conj(single_spectrum), order, (key1, key2))

    def array_prep(self, orders, f_all_in, dataset_idx):
        f_max_idx = f_all_in.shape[0]
        for order in orders:
            self.freq[dataset_idx][order] = f_all_in
            if order == 1:
                self.s_errs[dataset_idx][order] = torch.ones((1, self.sconfig.m_var), 
                                                              device=self.sconfig.backend,
                                                              dtype=torch.complex64)
            elif order == 2:
                self.s_errs[dataset_idx][order] = torch.ones((f_max_idx, self.sconfig.m_var), 
                                                              device=self.sconfig.backend,
                                                              dtype=torch.complex64)

            elif order == 4:
                self.s_errs[dataset_idx][order] = torch.ones((f_max_idx, f_max_idx, self.sconfig.m_var), # double check later
                                                              device=self.sconfig.backend,
                                                              dtype=torch.complex64)

    def process_order(self):
        return [1, 2, 4] if self.sconfig.order_in == 'all' else self.sconfig.order_in

    def reset_variables(self, orders, f_lists=None):
        self.err_counter = {i: {1: 0, 2: 0, 4: 0} for i in self.selected}
        self.err_counter.update({j: {2: 0, 4: 0} for j in self.cross2_selected})
        self.n_error_estimates = {i: {1: 0, 2: 0, 4: 0} for i in self.selected}
        self.n_error_estimates.update({j: {2: 0, 4: 0} for j in self.cross2_selected})
        for dataset_idx in self.selected:
            for order in orders:
                self.f_lists[dataset_idx][order] = f_lists
                self.freq[dataset_idx][order] = None
                self.s[dataset_idx][order] = None
                self.s_gpu[dataset_idx][order] = None
                self.s_err[dataset_idx][order] = None
                self.s_err_gpu[dataset_idx][order] = None
                self.s_errs[dataset_idx][order] = []
                self.m[order] = self.sconfig.m
                self.m_var[order] = self.sconfig.m_var

        for cross2_idx in self.cross2_selected:
            for order in orders:
                self.f_lists[cross2_idx][order] = f_lists
                self.freq[cross2_idx][order] = None
                self.s[cross2_idx][order] = None
                self.s_gpu[cross2_idx][order] = None
                self.s_err[cross2_idx][order] = None
                self.s_err_gpu[cross2_idx][order] = None
                self.s_errs[cross2_idx][order] = []

    def reset(self):
        orders = self.process_order()
        self.orders = orders
        self.reset_variables(orders, f_lists=self.sconfig.f_lists)
        return orders

    def setup_calc_spec(self, orders):
        f_max_allowed = 1 / (2 * self.sconfig.dt)
        if self.sconfig.f_max is None:
            self.sconfig.f_max = f_max_allowed

        window_len_factor = f_max_allowed / (self.sconfig.f_max - self.sconfig.f_min)
        self.t_window = (self.sconfig.spectrum_size - 1) * (2 * self.sconfig.dt * window_len_factor)

        n_data_points = self.diconfig_list[self.selected[0]].data.shape[0]
        window_points = int(np.round(self.t_window / self.sconfig.dt))

        if not window_points * self.sconfig.m + window_points // 2 < n_data_points:
            m = (n_data_points - window_points // 2) // window_points
            if m < max(orders):
                raise ValueError('Not enough data points')
            print(f'Values have been changed. Old m: {self.sconfig.m}, new m: {m}')
            self.sconfig.m = m
        else:
            m = self.sconfig.m

        denom_spec = window_points * m + window_points // 2
        n_spectra = n_data_points // denom_spec

        if n_spectra < self.sconfig.m:
            m_var = n_data_points // denom_spec
            if m_var < 2:
                raise ValueError('Not enough data points.')
            else:
                print(f'm_var values have been changed. Old: {self.sconfig.m_var}, new: {m_var}')
            self.m_var = m_var

        n_windows = int(np.floor(n_data_points / (m * window_points)))
        # Create frequency axis using full FFT if needed
        if self.use_full_fft:
            freq_all_freq = np.fft.fftfreq(window_points, self.sconfig.dt)
            freq_all_freq = np.fft.fftshift(freq_all_freq)
        else:
            freq_all_freq = np.fft.rfftfreq(window_points, self.sconfig.dt)

        # Determine indices for frequency band
        f_mask = freq_all_freq <= self.sconfig.f_max
        f_max_idx = np.sum(f_mask)
        f_mask = freq_all_freq < self.sconfig.f_min
        f_min_idx = np.sum(f_mask)

        return m, window_points, freq_all_freq, f_max_idx, f_min_idx, n_windows

    def _to_device(self, array):
        """Converts a NumPy array to a torch tensor on the proper device."""
        tensor = torch.from_numpy(array.astype(np.float32)) if self.use_float32 else torch.from_numpy(array)
        return tensor.to(self.device)

    def _compute_fft(self, window, chunk_gpu):
        """Computes the FFT (full or real) and applies scaling and shift if needed."""
        if self.use_full_fft:
            a_w = torch.fft.fft(window * chunk_gpu, dim=1)
            a_w *= self.sconfig.dt
            a_w = torch.fft.fftshift(a_w, dim=1)
        else:
            a_w = torch.fft.rfft(window * chunk_gpu, dim=1)
            a_w *= self.sconfig.dt
        return a_w

    def calc_spec(self):
        """
        Calculate spectra using PyTorch.
        """
        orders = self.reset()
        cross_orders = [2]
        m, window_points, freq_all_freq, f_max_idx, f_min_idx, n_windows = self.setup_calc_spec(orders)

        for order in orders:
            self.m[order] = m

        single_window, _ = cg_window(int(window_points), self.fs)
        window = np.array(m * [single_window]).flatten().reshape((m, window_points, 1))
        window = self._to_device(window)

        if self.sconfig.show_first_frame:
            self.plot_first_frames(self.selected, window_points)

        for dataset_idx in self.selected:
            self.array_prep(orders, freq_all_freq[f_min_idx:f_max_idx], dataset_idx)
        if self.cross2_selected:
            for pair in self.cross2_selected:
                self.array_prep(cross_orders, freq_all_freq[f_min_idx:f_max_idx], pair)

        for i in tqdm(range(n_windows), leave=False):
            for window_shift in [0, window_points // 2]:
                a_w_all_dict = {}
                for dataset_idx in self.selected:
                    data_config = self.diconfig_list[dataset_idx]
                    start = int(i * (window_points * m) + window_shift)
                    end = int((i + 1) * (window_points * m) + window_shift)
                    chunk = data_config.data[start:end]
                    if chunk.shape[0] == window_points * m:
                        chunk_r = chunk.reshape((m, window_points, 1))
                        chunk_gpu = self._to_device(chunk_r)
                        a_w_all_dict[dataset_idx] = self._compute_fft(window, chunk_gpu)
                    else:
                        a_w_all_dict[dataset_idx] = None

                if self.cconfig.auto_corr:
                    for dataset_idx, a_w_all_gpu in a_w_all_dict.items():
                        if a_w_all_gpu is None:
                            continue
                        self.n_chunks[dataset_idx] += 1
                        self.fourier_coeffs_to_spectra(orders, a_w_all_gpu, f_min_idx, f_max_idx, single_window, dataset_idx)
                        if self.n_chunks[dataset_idx] == self.sconfig.break_after:
                            break

                if self.cross2_selected:
                    for key1, key2 in self.cross2_selected:
                        if a_w_all_dict.get(key1) is None or a_w_all_dict.get(key2) is None:
                            continue
                        self.n_chunks[(key1, key2)] += 1
                        self.fourier_coeffs_to_cross_spectra(cross_orders, a_w_all_dict, f_min_idx, f_max_idx, single_window, key1, key2)
                        if self.n_chunks[(key1, key2)] == self.sconfig.break_after:
                            break

        for dataset_idx in self.selected:
            self.store_final_spectrum(orders, self.n_chunks[dataset_idx], dataset_idx)
        if self.cross2_selected:
            for key1, key2 in self.cross2_selected:
                self.store_final_spectrum(cross_orders, self.n_chunks[(key1, key2)], (key1, key2))

        return self.freq, self.s, self.s_err
