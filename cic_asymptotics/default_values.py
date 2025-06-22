DEFAULT_SAMPLE_SIZE = 1_000

DEFAULT_ESTIMATORS = {
    "Standard kernel": {"method": "standard", "se_method": "kernel"},
    "Standard Xavier": {"method": "standard", "se_method": "xavier"},
    "Smooth kernel": {"method": "smoothed", "se_method": "kernel"},
    "Smooth LS": {"method": "smoothed", "se_method": "lewbel-schennach"},
    "Smooth Xavier": {"method": "smoothed", "se_method": "xavier"},
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
