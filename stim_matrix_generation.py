import numpy as np
import pandas as pd
import stim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qldpc.codes import CSSCode, ClassicalCode
from qldpc.circuits import get_memory_experiment
from qldpc.circuits.noise_model import NoiseModel
from qldpc.objects import Pauli


class BitFlipNoiseModel(NoiseModel):
    """Pure X-error (bit-flip) noise model."""
    def __init__(self, p: float) -> None:
        self.p = p
        super().__init__(
            clifford_1q_error=p,
            clifford_2q_error=p,
            readout_error=p,
            reset_error=p,
        )


Hx = np.loadtxt("..\Simplified\H_Matrix\hx_ibm_72.txt", dtype=int)
Hz = np.loadtxt("..\Simplified\H_Matrix\hz_ibm_72.txt", dtype=int)

code = CSSCode(code_z=ClassicalCode(Hz), code_x=ClassicalCode(Hx))

p = 4e-3
aux = '4_3'
num_rounds = 6

noise_model = BitFlipNoiseModel(p)

circuit = get_memory_experiment(code, basis=Pauli.Z, num_rounds=num_rounds, noise_model=noise_model)
dem = circuit.detector_error_model(
    decompose_errors=True,
    approximate_disjoint_errors=True,
    ignore_decomposition_failures=True,
)
flat = dem.flattened()
print(flat)
print(len(flat))

error_instructions = [inst for inst in flat if inst.type == "error"]
num_errors = len(error_instructions)

H = np.zeros((flat.num_detectors, num_errors), dtype=np.uint8)
L = np.zeros((flat.num_observables, num_errors), dtype=np.uint8)
p_vec = np.zeros(num_errors, dtype=np.float64)

for col, inst in enumerate(error_instructions):
    p_vec[col] = inst.args_copy()[0]
    for target in inst.targets_copy():
        if target.is_relative_detector_id():
            H[target.val, col] ^= 1
        elif target.is_logical_observable_id():
            L[target.val, col] ^= 1

print(f"H shape: {H.shape}  ({H.shape[0]} rows / detectors, {H.shape[1]} cols / errors)")
print(f"L shape: {L.shape}  ({L.shape[0]} rows / observables, {L.shape[1]} cols / errors)")

np.savetxt(f"dem_H_{aux}.txt", H, fmt="%d")
np.savetxt(f"dem_L_{aux}.txt", L, fmt="%d")
np.savetxt(f"dem_p_{aux}.txt", p_vec[np.newaxis, :], fmt="%.6e")

sampler = circuit.compile_detector_sampler()
shots = sampler.sample(100000,append_observables=True)



detector_shots = shots[:, :flat.num_detectors]
logical_shots  = shots[:, flat.num_detectors:]

np.savetxt(f"detector_shots_{aux}.txt", detector_shots, fmt="%d")
np.savetxt(f"logical_shots_{aux}.txt",  logical_shots,  fmt="%d")

print(f"detector_shots: {detector_shots.shape}  ({detector_shots.shape[0]} shots, {detector_shots.shape[1]} detectors)")
print(f"logical_shots:  {logical_shots.shape}  ({logical_shots.shape[0]} shots, {logical_shots.shape[1]} logicals)")

