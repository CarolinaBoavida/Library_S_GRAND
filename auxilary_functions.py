from matplotlib import rc
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import csv
import ctypes


class GrandDecoderConfig(ctypes.Structure):
    _fields_ = [
        ("max_error_weight", ctypes.c_int),
        ("threads_per_block", ctypes.c_int),
        ("use_gpu_sum", ctypes.c_int),
        ("max_size_active_positions", ctypes.c_int),
        ("threshold_for_hybrid", ctypes.c_int),
        ("mode_of_search", ctypes.c_int),
    ]


class GrandDecoderResult(ctypes.Structure):
    _fields_ = [
        ("found", ctypes.c_int),
        ("found_weight", ctypes.c_int),
        ("original_weight", ctypes.c_int),
        ("degenerate", ctypes.c_int),
        ("is_equal", ctypes.c_int),
        ("tam", ctypes.c_int),
        ("total_tested_combinations", ctypes.c_ulonglong),
        ("hibrido", ctypes.c_int),
        ("iteration", ctypes.c_int),
        ("time_sum_ms", ctypes.c_double),
        ("total_time_search_ms", ctypes.c_double),
        ("transfer_time_ms", ctypes.c_double),
        ("total_to_find_ms", ctypes.c_double),
    ]

def save_syndromes_to_file(syndromes, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=" ")
        for syndrome in syndromes:
            writer.writerow(syndrome)


def read_errors_or_syndromes_from_file(file_path, n_cols, max_shots=None):
    errors_or_syndromes = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) == n_cols:
                error_or_syndrome = [int(x) for x in parts]

            elif len(parts) == 1 and len(parts[0]) == n_cols:
                error_or_syndrome = [int(x) for x in parts[0]]

            else:
                raise ValueError(
                    f"Erro com tamanho errado: esperado {n_cols}, "
                    f"recebido {len(parts)} tokens. Linha: {line[:100]}"
                )

            errors_or_syndromes.append(error_or_syndrome)

            if max_shots is not None and len(errors_or_syndromes) >= max_shots:
                break

    return np.array(errors_or_syndromes, dtype=np.int32)

def save_results_cc_csv(output_file, results, num_shots, using_errors):
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if(using_errors): 
            writer.writerow([ "shot", "found", "found_weight", "original_weight", "degenerate", "is_equal", "size_active_positions", "total_tested_combinations",
            "hybrid", "iteration", "time_sum_ms", "total_time_search_ms", "transfer_time_ms", "total_to_find_ms", ])
        else:
            writer.writerow([ "shot", "found", "found_weight", "size_active_positions", "total_tested_combinations",
            "hybrid", "iteration", "time_sum_ms", "total_time_search_ms", "transfer_time_ms", "total_to_find_ms", ])

        for i in range(num_shots):
            r = results[i]

            if(using_errors):
                writer.writerow([ i, r.found, r.found_weight, r.original_weight, r.degenerate, r.is_equal, r.tam, r.total_tested_combinations,
                r.hibrido, r.iteration, r.time_sum_ms, r.total_time_search_ms, r.transfer_time_ms, r.total_to_find_ms, ])
            else: 
                writer.writerow([ i, r.found, r.found_weight, r.tam, r.total_tested_combinations,
                r.hibrido, r.iteration, r.time_sum_ms, r.total_time_search_ms, r.transfer_time_ms, r.total_to_find_ms, ])
                
def save_results_cln_csv(output_file, results, num_shots, using_errors):
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        writer.writerow([ "shot", "found", "found_weight", "original_weight", "degenerate", "is_equal", "size_active_positions", "total_tested_combinations",
        "hybrid", "iteration", "time_sum_ms", "total_time_search_ms", "transfer_time_ms", "total_to_find_ms", ])
        
        for i in range(num_shots):
            r = results[i]

            writer.writerow([ i, r.found, r.found_weight, r.original_weight, r.degenerate, r.is_equal, r.tam, r.total_tested_combinations,
            r.hibrido, r.iteration, r.time_sum_ms, r.total_time_search_ms, r.transfer_time_ms, r.total_to_find_ms, ])
                

def read_matrix_from_file(file_path):
    rows = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = [int(x) for x in line.split()]
            rows.append(row)
            
    H = np.array(rows, dtype=np.int8)
    
    return H


def compute_syndrome(H, error):
    syndrome = (H @ error) % 2
    return syndrome

def simulate_random_error(n, p_error):
    while True:
        e = (np.random.rand(n) < p_error).astype(int)
        if np.any(e):  
            return e
        
def build_quantum_circuit_H (H, error_positions, measure):
    m, n = H.shape
    
    qc = QuantumCircuit(n + m, m) # tenho n qubits de dados e m qubits auxiliares (ancillas)
    
    data = list(range(n))  # índices dos qubits de dados
    ancillas = list(range(n, n + m))  # índices dos qubits auxilia
    
    if error_positions is not None:
        for q in error_positions:
            qc.x(data[q])
    
    for row in range(m):
        for col in range(n):
            if H[row, col] == 1:
                qc.cx(data[col], ancillas[row])  # CNOT do qubit de dados para o qubit auxiliar correspondente
                
    if measure:
        for row in range(m):
            qc.measure(ancillas[row], row)  # medir o qubit auxiliar e armazenar o resultado no registrador clássico correspondente
                
    return qc

def extract_syndrome(qc, n_shots):
    
    sim = AerSimulator()
    result = sim.run(qc, shots=n_shots).result()
    counts = result.get_counts()

    bitstring = next(iter(counts))
    syndrome = [int(b) for b in bitstring[::-1]]

    return syndrome

def error_positions_random(n, error_weight):
    positions = np.random.choice(n, size=error_weight, replace=False)
    return positions

def load_library(lib_path="./libgranddecoder.so"):
    lib = ctypes.CDLL(lib_path)

    lib.grand_decode_cc.argtypes = [
        ctypes.POINTER(ctypes.c_int),       # Hx
        ctypes.POINTER(ctypes.c_int),       # Hz
        ctypes.c_int,                       # n
        ctypes.c_int,                       # m
        ctypes.POINTER(ctypes.c_int),       # errors
        ctypes.POINTER(ctypes.c_int),       # syndromes
        ctypes.c_int,                       # num_shots
        ctypes.POINTER(GrandDecoderConfig), # config
        ctypes.POINTER(GrandDecoderResult), # results
        ctypes.c_char                       # type_of_error
    ]

    lib.grand_decode_cc.restype = ctypes.c_int
    
    lib.grand_decode_prob_cc.argtypes = [
        ctypes.POINTER(ctypes.c_int),       # Hx
        ctypes.POINTER(ctypes.c_int),       # Hz
        ctypes.c_int,                       # n
        ctypes.c_int,                       # m
        ctypes.POINTER(ctypes.c_int),       # errors
        ctypes.POINTER(ctypes.c_int),       # syndromes
        ctypes.c_int,                       # num_shots
        ctypes.POINTER(GrandDecoderConfig), # config
        ctypes.POINTER(GrandDecoderResult), # results                   
        ctypes.c_char,                      # type_of_error
        ctypes.c_char_p                     # prob_file_path
    ]

    lib.grand_decode_prob_cc.restype = ctypes.c_int

    return lib

def run_grand_shots_cc(lib, Hx, Hz, errors, syndromes, cfg, error_type):
   
    n_rows, n_cols = Hx.shape

    Hx_flat = np.ascontiguousarray(Hx.ravel(), dtype=np.int32)
    Hz_flat = np.ascontiguousarray(Hz.ravel(), dtype=np.int32)

    null_int_p = ctypes.POINTER(ctypes.c_int)()

    if errors is not None:
        num_shots = errors.shape[0]
        errors_flat = np.ascontiguousarray(errors.ravel(), dtype=np.int32)
        errors_ptr = errors_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        syndromes_ptr = null_int_p

    else:
        num_shots = syndromes.shape[0]
        syndromes_flat = np.ascontiguousarray(syndromes.ravel(), dtype=np.int32)
        errors_ptr = null_int_p
        syndromes_ptr = syndromes_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    results = (GrandDecoderResult * num_shots)()

    rc = lib.grand_decode_cc(
        Hx_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), Hz_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n_rows,n_cols, errors_ptr, syndromes_ptr, num_shots,
        ctypes.byref(cfg), results, ctypes.c_char(error_type),
    )

    return rc, results

def run_grand_prob_shots_cc(lib, Hx, Hz, errors, syndromes, cfg, prob_file_path, error_type):
    
    n_rows, n_cols = Hx.shape

    Hx_flat = np.ascontiguousarray(Hx.ravel(), dtype=np.int32)
    Hz_flat = np.ascontiguousarray(Hz.ravel(), dtype=np.int32)
    
    null_int_p = ctypes.POINTER(ctypes.c_int)()
    
    if errors is not None:
        num_shots = errors.shape[0]
        errors_flat = np.ascontiguousarray(errors.ravel(), dtype=np.int32)  
        errors_ptr = errors_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        syndromes_ptr = null_int_p

    else:
        num_shots = syndromes.shape[0]
        syndromes_flat = np.ascontiguousarray(syndromes.ravel(), dtype=np.int32)
        errors_ptr = null_int_p
        syndromes_ptr = syndromes_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    results = (GrandDecoderResult * num_shots)()

    rc = lib.grand_decode_prob_cc(
        Hx_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), Hz_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n_rows, n_cols, errors_ptr, syndromes_ptr,
        num_shots, ctypes.byref(cfg), results, ctypes.c_char(error_type), prob_file_path.encode('utf-8'),
    )

    return rc, results

def run_grand_shots_cln(lib, H_dem, L_dem, detectors, logicals, n_logicos, cfg):
    print("Running GRAND shots for CLN...")
    
