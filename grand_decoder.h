/**
 * @file grand_decoder.h
 * @brief Public API for the GRAND (Guessing Random Additive Noise Decoding)
 *        decoder, GPU-accelerated with CUDA, for quantum error correction.
 *
 * The library exposes decoders for two input representations:
 *  - CSS-style parity-check matrices (Hx / Hz), used by grand_decode_cc
 *    and grand_decode_prob_cc.
 *  - Detector-error-model (DEM) style matrices (H_dem / L_dem), used by
 *    grand_decode_cln, grand_decode_prob_cln and grand_decode_ML_cln.
 *
 * All decoding functions process a batch of `num_shots` syndromes/detectors
 * in a single call and report per-run statistics via GrandDecoderResult.
 */

#ifndef GRAND_DECODER_H
#define GRAND_DECODER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/** Maximum error weight (number of simultaneous errors) supported by the
 *  combinatorial search kernels. Increasing this requires re-sizing the
 *  internal combination lookup tables. */
#define GRAND_MAX_W 6

/**
 * @brief Configuration parameters controlling a decoding run.
 *
 * Passed by the caller to every `grand_decode_*` function. Values outside
 * the documented ranges are rejected by the decoder (an error is printed
 * to stderr and the call returns -1).
 */
typedef struct {
    int max_error_weight;          /**< Maximum error weight to search for. Must be in [1, 7]. */
    int threads_per_block;         /**< CUDA threads per block used by the search kernels. */
    int use_gpu_sum;                /**< Non-zero to compute column-weight sums on the GPU instead of the CPU. */
    int max_size_active_positions;  /**< Upper bound on the number of active (candidate) columns considered per shot. */
    int threshold_for_hybrid;       /**< Search-space size threshold above which hybrid mode switches from CPU to GPU search. */
    int mode_of_search;             /**< Search strategy: -1 = CPU-only search, 0 = GPU-only search, 1 = hybrid CPU/GPU search. */
} GrandDecoderConfig;

/**
 * @brief Result and timing statistics produced by a single decoding call.
 *
 * When `num_shots > 1`, these fields report aggregate/last-shot statistics
 * as implemented by the decoder (see the corresponding .cu source for the
 * exact aggregation semantics of each field).
 */
typedef struct {
    int found;                     /**< Non-zero if a matching error pattern was found. */
    int found_weight;              /**< Weight of the error pattern found (number of errors). */
    int original_weight;           /**< Weight of the true/injected error, when known (e.g. simulation ground truth). */
    int degenerate;                /**< Non-zero if a degenerate solution (equivalent alternative error) was detected. */
    int is_equal;                  /**< Non-zero if the found error matches the original error exactly. */
    int tam;                       /**< Size of the active search space (number of candidate positions) used. */
    unsigned long long total_tested_combinations; /**< Total number of error combinations tested during the search. */
    int hibrido;                   /**< Indicates whether hybrid mode triggered a switch to GPU search (implementation-defined encoding). */
    int iteration;                 /**< Number of search iterations/phases performed before termination. */

    double time_sum_ms;            /**< Time spent computing column-weight sums, in milliseconds. */
    double total_time_search_ms;   /**< Total time spent in the combinatorial search, in milliseconds. */
    double transfer_time_ms;       /**< Time spent on host-device memory transfers, in milliseconds. */
    double total_to_find_ms;       /**< Total wall-clock time to find (or exhaust the search for) an error, in milliseconds. */
} GrandDecoderResult;



/**
 * @brief GRAND decoder for CSS codes, using a per-position error probability model.
 *
 * Decodes `num_shots` syndromes against a CSS parity-check matrix pair
 * (Hx, Hz), biasing the combinatorial search order using per-qubit error
 * probabilities loaded from `probability_file`.
 *
 * @param Hx                X-type parity-check matrix (n x m, row-major, binary).
 * @param Hz                Z-type parity-check matrix (n x m, row-major, binary).
 * @param n                 Number of rows (checks) in Hx/Hz.
 * @param m                 Number of columns (qubits) in Hx/Hz.
 * @param errors            Output buffer for decoded error patterns (may be NULL if only syndromes are supplied; see implementation).
 * @param syndromes         Input syndromes, num_shots x n (row-major, binary).
 * @param num_shots         Number of shots (syndromes) to decode in this call.
 * @param config            Decoder configuration (search mode, thresholds, etc.).
 * @param results           Output array of num_shots result/statistics records.
 * @param type_of_error     Selects which error channel to decode: 'x'/'X' for X-type errors (uses Hz), 'z'/'Z' for Z-type errors (uses Hx).
 * @param probability_file  Path to a file with per-position error probabilities used to bias the search order.
 * @return 0 on success, negative value on error (invalid config, allocation failure, etc.).
 */
extern "C" int grand_decode_prob_cc(const int *Hx, const int *Hz, int n, int m, int *errors, int *syndromes, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results, char type_of_error, const char *probability_file);

/**
 * @brief GRAND decoder for CSS codes, using uniform (unweighted) error search order.
 *
 * Same as grand_decode_prob_cc but without probability weighting: error
 * weights are searched in plain increasing order.
 *
 * @param Hx             X-type parity-check matrix (n x m, row-major, binary).
 * @param Hz             Z-type parity-check matrix (n x m, row-major, binary).
 * @param n              Number of rows (checks) in Hx/Hz.
 * @param m              Number of columns (qubits) in Hx/Hz.
 * @param errors         Output buffer for decoded error patterns (may be NULL; see implementation).
 * @param syndromes      Input syndromes, num_shots x n (row-major, binary).
 * @param num_shots      Number of shots (syndromes) to decode in this call.
 * @param config         Decoder configuration (search mode, thresholds, etc.).
 * @param results        Output array of num_shots result/statistics records.
 * @param type_of_error  Selects which error channel to decode: 'x'/'X' for X-type errors (uses Hz), 'z'/'Z' for Z-type errors (uses Hx).
 * @return 0 on success, negative value on error.
 */
extern "C" int grand_decode_cc(const int *Hx, const int *Hz, int n, int m, int *errors, int *syndromes, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results, char type_of_error);

/**
 * @brief GRAND decoder for detector-error-model (DEM) style codes.
 *
 * Decodes `num_shots` detector outcomes against a DEM parity-check matrix
 * `H_dem` and reports which logical observables (columns of `L_dem`) are
 * flipped by the recovered error, using a uniform search order.
 *
 * @param H_dem       DEM parity-check matrix (n x m, row-major, binary).
 * @param L_dem       Logical-observable matrix (m x n_logicos, row-major, binary) mapping error mechanisms to logical flips.
 * @param n           Number of rows (detectors) in H_dem.
 * @param m           Number of columns (error mechanisms) in H_dem.
 * @param n_logicos   Number of logical observables (columns of L_dem).
 * @param detectors   Input detector outcomes, num_shots x n (row-major, binary).
 * @param logicals    Output buffer for the decoded logical-flip outcomes, num_shots x n_logicos.
 * @param num_shots   Number of shots to decode in this call.
 * @param config      Decoder configuration (search mode, thresholds, etc.).
 * @param results     Output array of num_shots result/statistics records.
 * @return 0 on success, negative value on error (e.g. invalid max_error_weight).
 */
extern "C" int grand_decode_cln(int *H_dem, const int *L_dem, int n, int m, int n_logicos, int *detectors, int *logicals, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results);

/**
 * @brief GRAND decoder for DEM-style codes, using a per-mechanism error probability model.
 *
 * Same as grand_decode_cln but biases the combinatorial search order using
 * per-error-mechanism probabilities loaded from `probability_file`.
 *
 * @param H_dem             DEM parity-check matrix (n x m, row-major, binary).
 * @param L_dem             Logical-observable matrix (m x n_logicos, row-major, binary).
 * @param n                 Number of rows (detectors) in H_dem.
 * @param m                 Number of columns (error mechanisms) in H_dem.
 * @param n_logicos         Number of logical observables (columns of L_dem).
 * @param detectors         Input detector outcomes, num_shots x n (row-major, binary).
 * @param logicals          Output buffer for the decoded logical-flip outcomes, num_shots x n_logicos.
 * @param num_shots         Number of shots to decode in this call.
 * @param config            Decoder configuration (search mode, thresholds, etc.).
 * @param results           Output array of num_shots result/statistics records.
 * @param probability_file  Path to a file with per-error-mechanism probabilities used to bias the search order.
 * @return 0 on success, negative value on error.
 */
extern "C" int grand_decode_prob_cln(int *H_dem, const int *L_dem, int n, int m, int n_logicos, int *detectors, int *logicals, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results, const char *probability_file);

/**
 * @brief GRAND decoder for DEM-style codes, using maximum-likelihood (LLR-based) selection.
 *
 * Same as grand_decode_cln, but among matching error patterns of the search
 * the one with the minimum total log-likelihood-ratio (LLR) cost, loaded
 * from `llr_file`, is selected as the winning solution.
 *
 * @param H_dem       DEM parity-check matrix (n x m, row-major, binary).
 * @param L_dem       Logical-observable matrix (m x n_logicos, row-major, binary).
 * @param n           Number of rows (detectors) in H_dem.
 * @param m           Number of columns (error mechanisms) in H_dem.
 * @param n_logicos   Number of logical observables (columns of L_dem).
 * @param detectors   Input detector outcomes, num_shots x n (row-major, binary).
 * @param logicals    Output buffer for the decoded logical-flip outcomes, num_shots x n_logicos.
 * @param num_shots   Number of shots to decode in this call.
 * @param config      Decoder configuration (search mode, thresholds, etc.).
 * @param results     Output array of num_shots result/statistics records.
 * @param llr_file    Path to a file with per-error-mechanism LLR costs used for ML selection among matches.
 * @return 0 on success, negative value on error.
 */
extern "C" int grand_decode_ML_cln(int *H_dem, const int *L_dem, int n, int m, int n_logicos, int *detectors, int *logicals, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results, const char *llr_file);


#ifdef __cplusplus
}
#endif

#endif
