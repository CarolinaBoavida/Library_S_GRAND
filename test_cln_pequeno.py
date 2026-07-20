import ctypes as ct
import numpy as np
import stim
from qldpc.codes import CSSCode, ClassicalCode
from qldpc.circuits import get_memory_experiment
from qldpc.circuits.noise_model import NoiseModel
from qldpc.objects import Pauli


class BitFlipNoiseModel(NoiseModel):
    def __init__(self, p: float) -> None:
        self.p = p
        super().__init__(
            clifford_1q_error=p,
            clifford_2q_error=p,
            readout_error=p,
            reset_error=p,
        )


# ==================== ctypes: structs e prototipos (a partir de grand_decoder.h) ====================

class GrandDecoderConfig(ct.Structure):
    _fields_ = [
        ("max_error_weight", ct.c_int),
        ("threads_per_block", ct.c_int),
        ("use_gpu_sum", ct.c_int),
        ("max_size_active_positions", ct.c_int),
        ("threshold_for_hybrid", ct.c_int),
        ("mode_of_search", ct.c_int),
    ]


class GrandDecoderResult(ct.Structure):
    _fields_ = [
        ("found", ct.c_int),
        ("found_weight", ct.c_int),
        ("original_weight", ct.c_int),
        ("degenerate", ct.c_int),
        ("is_equal", ct.c_int),
        ("tam", ct.c_int),
        ("total_tested_combinations", ct.c_ulonglong),
        ("hibrido", ct.c_int),
        ("iteration", ct.c_int),
        ("time_sum_ms", ct.c_double),
        ("total_time_search_ms", ct.c_double),
        ("transfer_time_ms", ct.c_double),
        ("total_to_find_ms", ct.c_double),
    ]


def load_library(lib_path):
    lib = ct.CDLL(lib_path)

    lib.grand_decode_cln.argtypes = [
        ct.POINTER(ct.c_int),   # H_dem
        ct.POINTER(ct.c_int),   # L_dem
        ct.c_int,               # n
        ct.c_int,               # m
        ct.c_int,               # n_logicos
        ct.POINTER(ct.c_int),   # detectors
        ct.POINTER(ct.c_int),   # logicals
        ct.c_int,               # num_shots
        ct.POINTER(GrandDecoderConfig),
        ct.POINTER(GrandDecoderResult),
    ]
    lib.grand_decode_cln.restype = ct.c_int

    return lib


def to_c_int_ptr(arr):
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    return arr, arr.ctypes.data_as(ct.POINTER(ct.c_int))


# ==================== configuracao ====================

LIB_PATH = "./libgranddecoder.so"

Hx = np.loadtxt("../Simplified/H_Matrix/hx_ibm_72.txt", dtype=int)
Hz = np.loadtxt("../Simplified/H_Matrix/hz_ibm_72.txt", dtype=int)

code = CSSCode(code_z=ClassicalCode(Hz), code_x=ClassicalCode(Hx))

p = 4e-3
num_rounds = 6
MAX_ERROR_WEIGHT = 6

NUM_SHOTS_TESTE = 5000   # teste pequeno, como combinamos

# ==================== gerar DEM (H_dem, L_dem, p_vec) ====================

noise_model = BitFlipNoiseModel(p)
circuit = get_memory_experiment(code, basis=Pauli.Z, num_rounds=num_rounds, noise_model=noise_model)
dem = circuit.detector_error_model(
    decompose_errors=True,
    approximate_disjoint_errors=True,
    ignore_decomposition_failures=True,
)
flat = dem.flattened()

error_instructions = [inst for inst in flat if inst.type == "error"]
num_errors = len(error_instructions)

H_dem = np.zeros((flat.num_detectors, num_errors), dtype=np.int32)
L_dem = np.zeros((flat.num_observables, num_errors), dtype=np.int32)

for col, inst in enumerate(error_instructions):
    for target in inst.targets_copy():
        if target.is_relative_detector_id():
            H_dem[target.val, col] ^= 1
        elif target.is_logical_observable_id():
            L_dem[target.val, col] ^= 1

n_detectors = flat.num_detectors
n_logicos = flat.num_observables

print(f"H_dem: {H_dem.shape}   L_dem: {L_dem.shape}   n_logicos: {n_logicos}")

# ==================== amostrar shots (com acesso ao erro real, para o peso) ====================

err_sampler = flat.compile_sampler()
det_data, obs_data, err_data = err_sampler.sample(shots=NUM_SHOTS_TESTE, return_errors=True)

peso_real = err_data.sum(axis=1)
sindrome_nao_nula = det_data.any(axis=1)

mask_decodificavel = sindrome_nao_nula & (peso_real <= MAX_ERROR_WEIGHT)
mask_sindrome_nula = ~sindrome_nao_nula
mask_peso_alto = sindrome_nao_nula & (peso_real > MAX_ERROR_WEIGHT)

n_total = NUM_SHOTS_TESTE
n_sindrome_nula = mask_sindrome_nula.sum()
n_peso_alto = mask_peso_alto.sum()
n_decodificaveis = mask_decodificavel.sum()

print(f"total={n_total}  sindrome_nula={n_sindrome_nula}  peso>6={n_peso_alto}  decodificaveis={n_decodificaveis}")

# erro logico "escondido" em sindrome nula -- verificar antes de assumir sucesso automatico
erros_logicos_em_sindrome_nula = obs_data[mask_sindrome_nula].any(axis=1).sum()
if erros_logicos_em_sindrome_nula > 0:
    print(f"[aviso] {erros_logicos_em_sindrome_nula} sindromes nulas com erro logico nao detetado.")

# ==================== chamar o decoder so nos shots decodificaveis ====================

detectors_batch = det_data[mask_decodificavel].astype(np.int32)
logicals_batch = obs_data[mask_decodificavel].astype(np.int32)

lib = load_library(LIB_PATH)

H_dem_arr, H_dem_ptr = to_c_int_ptr(H_dem.flatten())
L_dem_arr, L_dem_ptr = to_c_int_ptr(L_dem.flatten())
detectors_arr, detectors_ptr = to_c_int_ptr(detectors_batch.flatten())
logicals_arr, logicals_ptr = to_c_int_ptr(logicals_batch.flatten())

config = GrandDecoderConfig(
    max_error_weight=MAX_ERROR_WEIGHT,
    threads_per_block=32,
    use_gpu_sum=0,
    max_size_active_positions=100,
    threshold_for_hybrid=10,
    mode_of_search=1,
)

num_shots_decoder = int(n_decodificaveis)
results_array = (GrandDecoderResult * num_shots_decoder)()

rc = lib.grand_decode_cln(
    H_dem_ptr, L_dem_ptr,
    n_detectors, num_errors, n_logicos,
    detectors_ptr, logicals_ptr,
    num_shots_decoder,
    ct.byref(config),
    results_array,
)

if rc != 0:
    raise RuntimeError(f"Decoder failed with rc={rc}")

# ==================== agregar LER real (decoder + falhas automaticas) ====================

n_not_found = sum(1 for r in results_array if r.found == 0)
n_logical_error_no_decoder = sum(1 for r in results_array if r.found == 1 and r.degenerate == 0)

n_erros_operational = (
    n_peso_alto                          # falha automatica (peso real > 6)
    + erros_logicos_em_sindrome_nula     # falha automatica (sindrome nula mas erro logico real)
    + n_not_found                        # decoder nao encontrou
    + n_logical_error_no_decoder         # decoder encontrou, mas errado
)

ler_operational_real = n_erros_operational / n_total

print("\n==================== resumo ====================")
print(f"n_total (real, todos os shots gerados): {n_total}")
print(f"  - sindrome nula (auto):           {n_sindrome_nula}")
print(f"  - peso real > 6 (auto-falha):      {n_peso_alto}")
print(f"  - enviados ao decoder:            {n_decodificaveis}")
print(f"      not_found:                     {n_not_found}")
print(f"      encontrado mas errado:         {n_logical_error_no_decoder}")
print(f"erros logicos totais (operational):  {n_erros_operational}")
print(f"LER operacional REAL:                {ler_operational_real:.6e}")
