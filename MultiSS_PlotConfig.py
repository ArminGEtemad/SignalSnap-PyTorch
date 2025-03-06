class PlotConfig:
    def __init__(self, f_min, f_max, display_orders=None, significance=1, arcsinh_scale=(False, 0.02), plot_format=None):
	    self.display_orders = display_orders if display_orders is not None else [1, 2, 4]
	    self.plot_lims = (f_min, f_max)
	    self.significance = significance
	    self.arcsinh_scale = arcsinh_scale
	    self.plot_format = plot_format if plot_format is not None else ['re', 'im']