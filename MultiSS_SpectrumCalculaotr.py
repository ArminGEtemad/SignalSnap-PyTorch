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
    def __init__(self, sconfig: SpectrumConfig,
                       cconfig: CrossConfig, 
                       diconfig_list: list[DataImportConfig],
                       selected=None):
        self.sconfig = sconfig
        self.cconfig = cconfig
        self.diconfig_list = diconfig_list
        self.selected = selected

        # if none is selected show all
        if self.selected is None:
            self.selected = list(range(len(diconfig_list)))

        self.cross2_selected = self.cconfig.cross_corr_2 if hasattr(self.cconfig, 'cross_corr_2') and isinstance(self.cconfig.cross_corr_2, list) else []

        self.device = torch.device(self.sconfig.backend)
        self.t_window = None
        self.t_unit = unit_conversion(sconfig.f_unit)
        self.n_chunks = {
            **{i: 0 for i in self.selected},  # For individual datasets
            **{(key1, key2): 0 for key1, key2 in self.cross2_selected}  # For cross-correlation pairs
        }
        self.m = {1: None, 2: None}
        self.m_var = {1: None, 2: None}
        self.fs = 1 / self.sconfig.dt
        self.freq = {
            **{i: {2: None} for i in self.selected},
            **{j: {2: None} for j in self.cross2_selected}
        }
        self.f_lists = {
            **{i: {2: None, 3: None, 4: None} for i in self.selected},
            **{j: {2: None, 3: None, 4: None} for j in self.cross2_selected}
        }
        self.s = {
            **{i: {1: None, 2: None} for i in self.selected},
            **{j: {2: None} for j in self.cross2_selected}
        }
        self.s_gpu = {
            **{i: {1: None, 2: None} for i in self.selected},
            **{j: {2: None} for j in self.cross2_selected}
        }
        self.s_err = {
            **{i: {1: None, 2: None} for i in self.selected},
            **{j: {2: None} for j in self.cross2_selected}
        }
        self.s_err_gpu = {
            **{i: {1: None, 2: None} for i in self.selected},
            **{j: {2: None} for j in self.cross2_selected}
        }
        self.s_errs = {
            **{i: {1: None, 2: []} for i in self.selected},
            **{j: {2: []} for j in self.cross2_selected}
        }
        self.err_counter = {
            **{i: {1: 0, 2: 0} for i in self.selected},
            **{j: {2: 0} for j in self.cross2_selected}
        }
        self.n_error_estimates = {
            **{i: {1: 0, 2: 0} for i in self.selected},
            **{j: {2: 0} for j in self.cross2_selected}
        }
        self.validate_shapes() # crashing the program if the data are not equally long

        # insurring MPS backend precision support
        if self.sconfig.backend == 'mps':
            self.use_float32 = True
            print('MPS backend on Apple harware supports single precision.\n'
                  'Using float32 for all tensors!')
        else:
            self.use_float32 = False

    def validate_shapes(self):
        """
        making sure that all the imported data have the same size and shape
        """
        expected_shape = self.diconfig_list[self.selected[0]].data.shape[0]

        for i, data_config in enumerate(self.diconfig_list):
            data_shape = data_config.data.shape[0]
            if data_shape != expected_shape:
                raise ValueError('Imported data must have same length!')

    def plot_first_frames(self, selected, window_size):
        n_plots = len(selected)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
        if n_plots == 1:
            axes = [axes]

        for i, selected_idx in enumerate(selected):
            data_config = self.diconfig_list[selected_idx]
            chunk = data_config.data
            first_frame = chunk[:window_size]
            t = np.arange(len(first_frame)) * self.sconfig.dt
            axes[i].plot(t, first_frame)
            axes[i].set_xlim([0, t[-1]])
            axes[i].set_title(f'first frame for data {selected_idx + 1}')
            axes[i].set_xlabel('t / ('+ self.t_unit + ')')
            axes[i].set_ylabel('amplitude')

        plt.tight_layout()
        plt.show()

    # ---- calculating unbiased cumulants ----
    def c1(self, a_w):
        """
        first cumulant is calculated via:
        c1 = <a_w>
        """
        s1 = torch.mean(a_w, dim=0)
        return s1[0]

    def c2(self, a_w1, a_w2):
        """
        second cumulant for multi-variable can be calculated via:
        c2 = m/(m-1) (<a_w1.a_w2*> - <a_w1>.<a_w2*>)
        """
        a_w2_star = torch.conj(a_w2)
        term_1 = torch.mean(a_w1 * a_w2_star, dim=0)
        if self.sconfig.coherent:
            s2 = term_1
        else:
            factor = self.sconfig.m / (self.sconfig.m - 1)
            term_2 = torch.mean(a_w1, dim=0) * torch.mean(a_w2_star, dim=0)
            s2 = factor * (term_1 - term_2)
        return s2.squeeze(-1)
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

        self.err_counter[dataset_idx][order] += 1

        if self.err_counter[dataset_idx][order] % self.sconfig.m_var == 0:
            dim = 1 if order in [1, 2] else 2

            # Increment error estimates unconditionally
            self.n_error_estimates[dataset_idx][order] += 1

            factor = self.sconfig.m_var / (self.sconfig.m_var - 1)
            mean_squared = torch.mean(self.s_errs[dataset_idx][order]**2, dim=dim)
            squared_mean = torch.mean(self.s_errs[dataset_idx][order], dim=dim)**2
            s_err_gpu = factor * (mean_squared - squared_mean)
            s_err_gpu /= self.sconfig.m_var  # Corrected this line as well

            if self.s_err[dataset_idx][order] is None:
                self.s_err[dataset_idx][order] = s_err_gpu.cpu().numpy()
            else:
                self.s_err[dataset_idx][order] += s_err_gpu.cpu().numpy()
            self.err_counter[dataset_idx][order] = 0

    def store_final_spectrum(self, orders, n_chunks, dataset_idx):
        for order in orders:
            if order in self.s_gpu.get(dataset_idx, {}):
                if self.s_gpu[dataset_idx][order] is not None:
                    # Average the accumulated spectrum
                    self.s_gpu[dataset_idx][order] /= n_chunks
                    self.s[dataset_idx][order] = self.s_gpu[dataset_idx][order].cpu().resolve_conj().numpy()
                    # Compute the error estimate
                    self.s_err[dataset_idx][order] = (
                        (1 / self.n_error_estimates[dataset_idx][order]) * np.sqrt(self.s_err[dataset_idx][order])
                    )
                    self.s_err[dataset_idx][order] /= 2  # for interlaced calculation
                else:
                    # s_gpu[dataset_idx][order] is None, skip processing
                    pass
            else:
                # The order doesn't exist for this dataset_idx, skip processing
                pass

    def fourier_coeffs_to_spectra(self, orders, coeffs_gpu, f_min_idx, f_max_idx, single_window, dataset_idx):
        for order in orders:
            if order == 1:
                a_w = coeffs_gpu[:, f_min_idx:f_max_idx, :]
                single_spectrum = self.c1(a_w) / (self.sconfig.dt *
                                                  single_window.mean() *
                                                  single_window.shape[0])
            elif order == 2:
                if self.sconfig.f_lists is None:
                    a_w = coeffs_gpu[:, f_min_idx:f_max_idx, :]
                else:
                    a_w = coeffs_gpu
                single_spectrum = self.c2(a_w, a_w) / (self.sconfig.dt * (single_window**2).sum())

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
                single_spectrum = self.c2(a_w1, a_w2) / (self.sconfig.dt * (single_window**2).sum())

            self.store_sum_single_spectrum(torch.conj(single_spectrum), order, (key1, key2))

    def array_prep(self, orders, f_all_in, dataset_idx):
        f_max_idx = f_all_in.shape[0]
        for order in orders:
            self.freq[dataset_idx][order] = f_all_in
            if order == 1:
                self.s_errs[dataset_idx][order] = torch.ones(
                    (1, self.sconfig.m_var), device=self.sconfig.backend,
                    dtype=torch.complex64)
            elif order == 2:
                self.s_errs[dataset_idx][order] = torch.ones(
                    (f_max_idx, self.sconfig.m_var),
                    device=self.sconfig.backend,
                    dtype=torch.complex64)

    def process_order(self):
        if self.sconfig.order_in == 'all':
            return [1, 2]
        else:
            return self.sconfig.order_in

    def reset_variables(self, orders, f_lists=None):
        self.err_counter = {
            **{i: {1: 0, 2: 0} for i in self.selected},
            **{j: {2: 0} for j in self.cross2_selected}
        }
        self.n_error_estimates = {
            **{i: {1: 0, 2: 0} for i in self.selected},
            **{j: {2: 0} for j in self.cross2_selected}
        }        
        for dataset_idx in self.selected:
            for order in orders:
                self.f_lists[order] = f_lists
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

        # since all data are equally long we can just take the shape of one of them
        n_data_points = self.diconfig_list[self.selected[0]].data.shape[0] 

        window_points = int(np.round(self.t_window / self.sconfig.dt))

        if not window_points * self.sconfig.m + window_points // 2 < n_data_points:
            m = (n_data_points - window_points // 2) // window_points
            if m < max(orders):
                max_spec_size = window_points // (2 * window_len_factor) + 1
                raise ValueError('Not enough data points')
            print(f'values have been changed. old: m = {self.m}, new m = {m}')
            self.sconfig.m = m
        else:
            m = self.sconfig.m

        denom_spec = window_points * m + window_points // 2
        n_spectra = n_data_points // denom_spec

        if  n_spectra < self.sconfig.m:
            m_var = n_data_points // denom_spec
            if m_var < 2:
                raise ValueError('Not enough data points.')
            else:
                print(f'm_var values have been changed. '
                      f'old:{self.sconfig.m_var}, new: {m_var}')
            self.m_var = m_var

        n_windows = int(np.floor(n_data_points / (m * window_points)))
        freq_all_freq = rfftfreq(int(window_points), self.sconfig.dt)
        # Is f_max too high? And find the f_max index
        f_mask = freq_all_freq <= self.sconfig.f_max
        f_max_idx = sum(f_mask)
        # Find the f_min index
        f_mask = freq_all_freq < self.sconfig.f_min
        f_min_idx = sum(f_mask)

        return m, window_points, freq_all_freq, f_max_idx, f_min_idx, n_windows

    def calc_spec(self):
        """
        calculating spectra using pytorch 
        """
        orders = self.reset()
        # Set cross_orders to only [2] for cross-correlations
        cross_orders = [2]

        m, window_points, freq_all_freq, f_max_idx, f_min_idx, n_windows = (
            self.setup_calc_spec(orders)
        )

        for order in orders:
            self.m[order] = m

        single_window, _ = cg_window(int(window_points), self.fs)
        window = np.array(m * [single_window]).flatten()
        window = window.reshape((m, window_points, 1))

        if self.use_float32:
            window = torch.from_numpy(window.astype(np.float32)).to(self.device)
        else:
            window = torch.from_numpy(window).to(self.device)

        if self.sconfig.show_first_frame:
            self.plot_first_frames(self.selected, window_points)

        data_configs = [self.diconfig_list[dataset_idx] for dataset_idx in self.selected]
        for dataset_idx in self.selected:
            self.array_prep(orders, freq_all_freq[f_min_idx:f_max_idx], dataset_idx)

        if self.cross2_selected is not None:
            for pair in self.cross2_selected:
                self.array_prep(cross_orders, freq_all_freq[f_min_idx:f_max_idx], pair)

        for i in tqdm(np.arange(0, n_windows, 1), leave=False):
            shift_iter = [0, window_points // 2]

            for window_shift in shift_iter:
                a_w_all_dict = {}  # Dictionary to store Fourier-transformed data

                for dataset_idx in self.selected:
                    data_config = self.diconfig_list[dataset_idx]

                    starting_idx = int(i * (window_points * m) + window_shift)
                    ending_idx = int((i + 1) * (window_points * m) + window_shift)

                    chunk = data_config.data[starting_idx:ending_idx]

                    # Validate and process chunk
                    if chunk.shape[0] == window_points * m:
                        chunk_r = chunk.reshape((m, window_points, 1))

                        # Transfer chunk to GPU
                        if self.use_float32:
                            chunk_gpu = torch.from_numpy(chunk_r.astype(np.float32)).to(self.device)
                        else:
                            chunk_gpu = torch.from_numpy(chunk_r).to(self.device)

                        # Perform Fourier Transform
                        a_w_all_gpu = torch.fft.rfft(window * chunk_gpu, dim=1)
                        a_w_all_gpu *= self.sconfig.dt  # Scale correction

                        # Store Fourier-transformed data in the dictionary
                        a_w_all_dict[dataset_idx] = a_w_all_gpu
                    else:
                        a_w_all_dict[dataset_idx] = None 

                # Process auto-correlation (individual datasets)
                if self.cconfig.auto_corr:
                    for dataset_idx, a_w_all_gpu in a_w_all_dict.items():
                        if a_w_all_gpu is None:  # Skip invalid data
                            continue

                        self.n_chunks[dataset_idx] += 1

                        self.fourier_coeffs_to_spectra(
                            orders, a_w_all_gpu, f_min_idx, f_max_idx, single_window, dataset_idx)

                        # Break if the dataset reaches its processing limit
                        if self.n_chunks[dataset_idx] == self.sconfig.break_after:
                            break

                # Process cross-correlations
                if self.cross2_selected is not None:
                    for pair in self.cross2_selected:
                        key1, key2 = pair

                        # Skip if either dataset is invalid
                        if a_w_all_dict.get(key1) is None or a_w_all_dict.get(key2) is None:
                            continue

                        self.n_chunks[(key1, key2)] += 1

                        # Call the cross-spectra function with cross_orders=[2]
                        self.fourier_coeffs_to_cross_spectra(
                            cross_orders, a_w_all_dict, f_min_idx, f_max_idx, single_window, key1, key2)

                        # Stop processing if chunk limit is reached
                        if self.n_chunks[(key1, key2)] == self.sconfig.break_after:
                            break

        # Store the final spectrum for each dataset or pair
        for dataset_idx in self.selected:
            self.store_final_spectrum(orders, self.n_chunks[dataset_idx], dataset_idx)

        if self.cross2_selected is not None:
            for pair in self.cross2_selected:
                key1, key2 = pair
                self.store_final_spectrum(cross_orders, self.n_chunks[(key1, key2)], (key1, key2))

        return self.freq, self.s, self.s_err
