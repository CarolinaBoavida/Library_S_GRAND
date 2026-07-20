import numpy as np
from auxilary_functions import *

HX_FILE = "H_Matrix/hx_ibm_72.txt"
HZ_FILE = "H_Matrix/hz_ibm_72.txt"


ERRORS_FILE = "Errors/patterns_72.txt"    # can be None if you only have syndromes and want to decode them
# SYNDROMES_FILE = "Syndromes/syndromes_72.txt" # can be None if you only have errors and want to decode them

NUM_SHOTS = 30


Hx = read_matrix_from_file(HX_FILE).astype(np.int32)
Hz = read_matrix_from_file(HZ_FILE).astype(np.int32)
n_rows, n_cols = Hx.shape

errors = read_errors_or_syndromes_from_file(ERRORS_FILE, n_cols=n_cols, max_shots=NUM_SHOTS)

syndromes = []

for error in errors:
    syndrome = compute_syndrome(Hz, error)      # i am using Hz because it is for X errors.
    # print(f"Error: {error}, Syndrome: {syndrome}")
    syndrome = np.asarray(syndrome, dtype=np.int32)
    syndromes.append(syndrome)
    
syndromes = np.array(syndromes, dtype=np.int32)

save_syndromes_to_file(syndromes, "Syndromes/syndromes_72.txt")
    


