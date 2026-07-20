import numpy as np
from auxilary_functions import *


H_DEM_FILE = "H_Matrix_DEM/dem_H_5_4.txt" # p = 5 x 10^-4
L_DEM_FILE = "L_Matrix_DEM/dem_L_5_4.txt"

DETECTOR_SHOTS_FILE = "Syndromes_DEM/detector_shots_5_4.txt"
LOGICAL_SHOTS_FILE = "Syndromes_DEM/logical_shots_5_4.txt"


OUTPUT_HYBRID_CSV = "results_hybrid_cln.csv"
# OUTPUT_HYBRID_PROB_CSV = "results_hybrid_prob.csv"

# PROB_FILE_PATH = "Probabilities/probabilities_72.txt"


LIB_PATH = "./libgranddecoder.so"

ERROR_TYPE = b"z"   # must be b"x"/b"X" for X errors, b"z"/b"Z" for Z errors
NUM_SHOTS = 30

MAX_ERROR_WEIGHT = 6     #   must be between 1 and 7, inclusive.    
THREADS_PER_BLOCK = 32   # try to use multiples of 32 for better performance, but it can be any positive integer.
USE_GPU_SUM = 1          # = 0 for CPU sum, = 1 for GPU sum
MAX_SIZE_ACTIVE_POSITIONS = 100
THRESHOLD_FOR_HYBRID = 10
MODE_OF_SEARCH = 1 #        -1 = cpu_search       ,       0 = gpu_search,             1 = hybrid_search

#------------------------------------------------------------------------------------------------------------

H_dem = read_matrix_from_file(H_DEM_FILE).astype(np.int32)
L_dem = read_matrix_from_file(L_DEM_FILE).astype(np.int32)

n_rows, n_cols = H_dem.shape


detectors = None
logicals = None

if DETECTOR_SHOTS_FILE is None and LOGICAL_SHOTS_FILE is None:
    raise ValueError("You must provide both detector shots and logical shots.")


if DETECTOR_SHOTS_FILE is not None:
    detectors = read_errors_or_syndromes_from_file(DETECTOR_SHOTS_FILE, n_cols=n_rows, max_shots=NUM_SHOTS)
    
if LOGICAL_SHOTS_FILE is not None:
    logicals = read_errors_or_syndromes_from_file(LOGICAL_SHOTS_FILE, n_cols=n_cols, max_shots=NUM_SHOTS)


if detectors is not None:
    num_shots = detectors.shape[0]

if logicals is not None:
    n_logicos = logicals.shape[1]


cfg = GrandDecoderConfig(
    max_error_weight=MAX_ERROR_WEIGHT,
    threads_per_block=THREADS_PER_BLOCK,
    use_gpu_sum=USE_GPU_SUM,
    max_size_active_positions=MAX_SIZE_ACTIVE_POSITIONS,
    threshold_for_hybrid=THRESHOLD_FOR_HYBRID,
    mode_of_search=MODE_OF_SEARCH
)

lib = load_library(LIB_PATH)

rc, results_hybrid = run_grand_shots_cln(lib=lib, H_dem=H_dem, L_dem=L_dem, detectors=detectors, logicals=logicals, n_logicos=n_logicos, cfg=cfg)

#rc, results_hybrid_prob = run_grand_prob_shots_cln(lib=lib, Hx=Hx, Hz=Hz, errors=errors, syndromes=syndromes, cfg=cfg, prob_file_path=PROB_FILE_PATH, error_type=ERROR_TYPE)

if rc != 0:
    raise RuntimeError(f"Decoder failed with rc={rc}")

save_results_cc_csv(OUTPUT_HYBRID_CSV, results_hybrid, num_shots=num_shots)
#save_results_cc_csv(OUTPUT_HYBRID_PROB_CSV, results_hybrid_prob, num_shots=num_shots, using_errors=using_errors)

print(f"Decoded {num_shots} shots")
#print(f"Results saved in: {OUTPUT_HYBRID_PROB_CSV}")
print(f"Results saved in: {OUTPUT_HYBRID_CSV}")