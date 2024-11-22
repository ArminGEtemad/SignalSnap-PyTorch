from MultiSS_SpectrumCalculaotr import SpectrumCalculator
from MultiSS_SpectrumConfig import SpectrumConfig, DataImportConfig
from MultiSS_CrossConfig import CrossConfig
from MultiSS_PlotConfig import PlotConfig

import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

class SpectrumPlotter:
	def __init__(self, sconfig: SpectrumConfig, cconfig: CrossConfig, scalc: SpectrumCalculator,  pconfig: PlotConfig):
		self.sconfig = sconfig
		self.scalc = scalc
		self.cconfig = cconfig
		self.pconfig = pconfig

	def import_spec_data(self, order, keys):
		s_data = self.scalc.s[keys][order].copy()
		s_err_data = self.scalc.s_err[keys][order].copy()
		freq_data = self.scalc.freq[keys][order].copy()

		return s_data, s_err_data, freq_data

	def signif_calculate(self, s_data, s_err_data):
	    """
	    Calculates the significance bounds for the given spectrum data.
	    """
	    return [
	        [s_data + (i + 1) * s_err_data for i in range(self.pconfig.significance)],
	        [s_data - (i + 1) * s_err_data for i in range(self.pconfig.significance)]
	    ]

	def display_s1(self, order, keys, source):
	    """
	    Function to handle the processing and display for order 1.
	    """
	    all_results = []
	    s_data, s_err_data, _ = self.import_spec_data(order, keys)

	    if s_data is not None and s_err_data is not None:
	        # Create a list of dictionaries for each row
	        spectrum = s_data
	        error_estimate = s_err_data

	        for i in range(len(spectrum)):
	            all_results.append({
	                'Dataset Index': keys,
	                'S1': spectrum[i].real,
	                'Error S1': error_estimate[i].real
	            })

	    return all_results

	def display_s2(self, order, datasets):
	    """
	    Function to handle plotting for order 2. Displays real and/or imaginary parts based on plot_format,
	    with significance bounds shaded in gray and the area between filled.
	    """
	    def plot_data(ax, freq_data, s_data, signif_bounds, component, label_prefix):
	        """
	        Helper to plot the data for the specified component (real/imag).
	        """
	        # Plot main data
	        data = getattr(s_data, component)
	        ax.plot(freq_data, data, label=f'{label_prefix} ({component.capitalize()})')

	        # Plot significance bounds
	        num_bounds = len(signif_bounds[0])
	        grays = np.linspace(0.8, 0.3, num_bounds)  # Gradual shades of gray
	        for i, (upper, lower) in enumerate(zip(signif_bounds[0], signif_bounds[1])):
	            upper_data = getattr(upper, component)
	            lower_data = getattr(lower, component)
	            ax.plot(freq_data, upper_data, linestyle='--', color=str(grays[i]), alpha=0.7)
	            ax.plot(freq_data, lower_data, linestyle='--', color=str(grays[i]), alpha=0.7)
	            ax.fill_between(
	                freq_data, lower_data, upper_data,
	                color=str(grays[i]), alpha=0.2,
	                label='Significance Bounds' if i == 0 else None
	            )

	    def configure_axes_s2(ax, title, ylabel):
	        """
	        Helper to configure axis labels and title.
	        """
	        ax.set_title(title)
	        ax.set_ylabel(ylabel)
	        ax.set_xlim(self.pconfig.plot_lims[0], self.pconfig.plot_lims[1])
	        ax.legend()

	    num_columns = len(self.pconfig.plot_format)
	    num_datasets = len(datasets)

	    # Create subplots with the required number of columns
	    fig, axes = plt.subplots(
	        num_datasets, num_columns, figsize=(8 * num_columns, 4 * num_datasets), sharex=True
	    )

	    # Normalize axes for consistent handling
	    if num_datasets == 1:
	        axes = [axes] if num_columns == 1 else [axes]
	    elif num_columns == 1:
	        axes = [[ax] for ax in axes]

	    for (keys, source), ax_row in zip(datasets, axes):
	        s_data, s_err_data, freq_data = self.import_spec_data(order, keys)

	        if s_data is not None and freq_data is not None and s_err_data is not None:
	            signif_bounds = self.signif_calculate(s_data, s_err_data)
	            for col, ax in enumerate(ax_row):
	                component = 'real' if self.pconfig.plot_format[col] == 're' else 'imag'
	                plot_data(ax, freq_data, s_data, signif_bounds, component, f'{source}: Dataset {keys}')
	                ylabel = f'{component.capitalize()} S Data'
	                title = f'{source}: {component.capitalize()} Part - Dataset {keys}'
	                configure_axes_s2(ax, title, ylabel)
	        else:
	            for ax in ax_row:
	                ax.text(0.5, 0.5, f"No data for {source}: key {keys}", ha='center', va='center', transform=ax.transAxes)
	                ax.set_title(f'{source}: Dataset {keys}')
	                ax.set_ylabel('S Data')
	                ax.set_xlim(self.pconfig.plot_lims[0], self.pconfig.plot_lims[1])

	    # Set shared x-axis label
	    if num_datasets == 1 and num_columns == 1:
	        axes[0][0].set_xlabel('Frequency')
	    else:
	        for ax in axes[-1]:
	            ax.set_xlabel('Frequency')

	    plt.tight_layout()
	    plt.show()

	def display(self):
	    all_results = []
	    datasets = []
	    generate_s2_plots = False

	    # Process datasets from both selected and cross2_selected
	    for source, selected_keys, valid_orders in [
	        ("selected", self.scalc.selected, [1, 2]),
	        ("cross2_selected", self.scalc.cross2_selected, [2]),
	    ]:
	        for keys in selected_keys:
	            for order in valid_orders:
	                if order == 1:
	                    results = self.display_s1(order, keys, source)
	                    all_results.extend(results)
	                elif order == 2:
	                    datasets.append((keys, source))
	                    generate_s2_plots = True

	    # Display the S1 results as a table
	    df = pd.DataFrame(all_results)
	    if not df.empty:
	        print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
	    else:
	        print("No results available for order 1.")

	    # Then handle the S2 plots
	    if generate_s2_plots:
	        self.display_s2(order=2, datasets=datasets)





