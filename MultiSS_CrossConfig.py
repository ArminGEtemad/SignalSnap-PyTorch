class InvalidConfigError(Exception):
    pass

class CrossConfig:
    def __init__(self, auto_corr=True, cross_corr_2=None, cross_corr_3=None, cross_corr_4=None):
        self.auto_corr = auto_corr
        self.cross_corr_2 = cross_corr_2
        self.cross_corr_3 = cross_corr_3
        self.cross_corr_4 = cross_corr_4

        self.validate()

    def validate(self):
        if not isinstance(self.auto_corr, (bool)):
            raise InvalidConfigError(f"Invalid 'auto_corr': {self.auto_corr}.\n"
                                     f"Must be boolian.")
