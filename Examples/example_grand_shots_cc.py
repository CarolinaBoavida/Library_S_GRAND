import numpy as np
from Examples.auxilary_functions import *


HX_FILE = "H_Matrix/hx_ibm_72.txt"
HZ_FILE = "H_Matrix/hz_ibm_72.txt"


ERRORS_FILE = None    # can be None if you only have syndromes and want to decode them
SYNDROMES_FILE = "Syndromes/syndromes_72.txt" # can be None if you only have errors and want to decode them

# if you have both errors and syndromes, the error vectors will be used. 

OUTPUT_HYBRID_CSV = "results_hybrid.csv"
OUTPUT_HYBRID_PROB_CSV = "results_hybrid_prob.csv"

PROB_FILE_PATH = "Probabilities/probabilities_72.txt"


LIB_PATH = "./libgranddecoder.so"

ERROR_TYPE = b"x"   # must be b"x"/b"X" for X errors, b"z"/b"Z" for Z errors
NUM_SHOTS = 30

MAX_ERROR_WEIGHT = 6     #   must be between 1 and 7, inclusive.    
THREADS_PER_BLOCK = 32   # try to use multiples of 32 for better performance, but it can be any positive integer.
USE_GPU_SUM = 0          # = 0 for CPU sum, = 1 for GPU sum
MAX_SIZE_ACTIVE_POSITIONS = 300
THRESHOLD_FOR_HYBRID = 10
MODE_OF_SEARCH = 1 #        -1 = cpu_search       ,       0 = gpu_search,             1 = hybrid_search

#------------------------------------------------------------------------------------------------------------

Hx = read_matrix_from_file(HX_FILE).astype(np.int32)
Hz = read_matrix_from_file(HZ_FILE).astype(np.int32)

n_rows, n_cols = Hx.shape

errors = None
syndromes = None

if ERRORS_FILE is not None:
    errors = read_errors_or_syndromes_from_file(ERRORS_FILE, n_cols=n_cols, max_shots=NUM_SHOTS)
    using_errors=1
else: 
    if SYNDROMES_FILE is not None:
        syndromes = read_errors_or_syndromes_from_file(SYNDROMES_FILE, n_cols=n_rows, max_shots=NUM_SHOTS)
        using_errors=0

if errors is not None and syndromes is not None:
    syndromes = None
    using_errors=1
    
if errors is None and syndromes is None:
    raise ValueError("You must provide either errors or syndromes.")

if errors is not None:
    num_shots = errors.shape[0]
else:
    num_shots = syndromes.shape[0]

cfg = GrandDecoderConfig(
    max_error_weight=MAX_ERROR_WEIGHT,
    threads_per_block=THREADS_PER_BLOCK,
    use_gpu_sum=USE_GPU_SUM,
    max_size_active_positions=MAX_SIZE_ACTIVE_POSITIONS,
    threshold_for_hybrid=THRESHOLD_FOR_HYBRID,
    mode_of_search=MODE_OF_SEARCH
)

lib = load_library(LIB_PATH)

rc, results_hybrid = run_grand_shots_cc(lib=lib, Hx=Hx, Hz=Hz, errors=errors, syndromes=syndromes, cfg=cfg, error_type=ERROR_TYPE)

rc, results_hybrid_prob = run_grand_prob_shots_cc(lib=lib, Hx=Hx, Hz=Hz, errors=errors, syndromes=syndromes, cfg=cfg, prob_file_path=PROB_FILE_PATH, error_type=ERROR_TYPE)

if rc != 0:
    raise RuntimeError(f"Decoder failed with rc={rc}")

save_results_cc_csv(OUTPUT_HYBRID_CSV, results_hybrid, num_shots=num_shots, using_errors=using_errors)
save_results_cc_csv(OUTPUT_HYBRID_PROB_CSV, results_hybrid_prob, num_shots=num_shots, using_errors=using_errors)

print(f"Decoded {num_shots} shots")
print(f"Results saved in: {OUTPUT_HYBRID_PROB_CSV}")
print(f"Results saved in: {OUTPUT_HYBRID_CSV}")