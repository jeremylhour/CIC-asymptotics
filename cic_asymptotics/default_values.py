DEFAULT_SAMPLE_SIZE = 1_000

DEFAULT_ESTIMATORS = {
    "Standard kernel": {"cdf_method": "standard", "se_method": "kernel"},
    "Standard Xavier": {"cdf_method": "standard", "se_method": "xavier"},
    "Smoothed kernel": {"cdf_method": "smoothed", "se_method": "kernel"},
    "Smoothed LS": {"cdf_method": "smoothed", "se_method": "lewbel-schennach"},
    "Smoothed Xavier": {"cdf_method": "smoothed", "se_method": "xavier"},
    "Smoothed Sample-splitting": {
        "cdf_method": "smoothed",
        "se_method": "sample-splitting",
    },
}

DEFAULT_CONFIG = {
    "dgp": "exponential",
    "n": DEFAULT_SAMPLE_SIZE,
    "params": {
        "lambda_x": 0.8,
        "lambda_z": 1,
        "alpha_y": 10,
    },
}

DEFAULT_SIMULATION_SIZE = 100

OUTPATH = "output/"
