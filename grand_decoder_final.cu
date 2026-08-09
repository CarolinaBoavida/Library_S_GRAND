#include "grand_decoder_final.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#define GRAND_MAX_MASKS 20

static const int INTERNAL_MAX_N = 300;

typedef struct {
    int tam_upper;
    int phase_order[3];
    float phase_prob[3];
} PhaseOrderMask;

static unsigned long long nCr_table[INTERNAL_MAX_N + 1][GRAND_MAX_W + 1];

__constant__ unsigned long long d_nCr_table[INTERNAL_MAX_N + 1][GRAND_MAX_W + 1];
__device__ int d_found_error_idx = -1;
__device__ int d_winning_combo[GRAND_MAX_W];

// -------- grand_decode_ML_cln: reducao por custo LLR minimo --------
__device__ unsigned long long d_best_key_ml = 0xFFFFFFFFFFFFFFFFULL;

__device__ void atomicMinULL_ml(unsigned long long *addr, unsigned long long val) {
    unsigned long long old = *addr, assumed;
    do {
        assumed = old;
        if (val >= assumed) return;
        old = atomicCAS(addr, assumed, val);
    } while (assumed != old);
}

// ------------------------------------------------ COMBINAÇÕES ----------------------------------------------------------------------------------


static unsigned long long nCr_cpu(int n, int k) {
    
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;

    unsigned long long res = 1;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - i + 1) / i;
    }
    return res;
}

static void setup_nCr_table_internal() {
    
    for (int n = 0; n <= INTERNAL_MAX_N; n++) {
        for (int k = 0; k <= GRAND_MAX_W; k++) {
            nCr_table[n][k] = nCr_cpu(n, k);
        }
    }

    cudaError_t err = cudaMemcpyToSymbol(d_nCr_table, nCr_table, sizeof(nCr_table));
}

// ------------------------------------------------ GPU ----------------------------------------------------------------------------------

__global__ void sum_columns_H_gpu( const int* d_H, const int* d_active_rows, int num_active_rows, int* d_sum_cols, int m) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < m) {
        int acc = 0;
        for (int i = 0; i < num_active_rows; i++) {
            int row = d_active_rows[i];
            if (d_H[row * m + col] == 1) {
                acc++;
            }
        }
        d_sum_cols[col] = acc;
    }
}

__device__ void get_combination( unsigned long long rank, int n, int k, int *out_combo ) {
    
    int candidate = 0;

    for (int i = 0; i < k; ++i) {
        int next_k = k - 1 - i;
        unsigned long long count = d_nCr_table[n - 1 - candidate][next_k];

        while (rank >= count && (n - 1 - candidate) > next_k) {
            rank -= count;
            candidate++;
            count = d_nCr_table[n - 1 - candidate][next_k];
        }

        out_combo[i] = candidate;
        candidate++;
    }
}

__global__ void search_errors_massively_prob( int *H, int *s, int *d_active_pos, int tam, int n_rows, int m_cols, int *d_phase_weights, int num_phase_weights) {
    
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long acc = 0;
    int selected_weight = -1;
    unsigned long long rank = 0;

    for (int i = 0; i < num_phase_weights; i++) {
        int w = d_phase_weights[i];
        unsigned long long cnt = d_nCr_table[tam][w];

        if (tid < acc + cnt) {
            selected_weight = w;
            rank = tid - acc;
            break;
        }

        acc += cnt;
    }

    if (selected_weight == -1) return;

    int combo[GRAND_MAX_W];
    get_combination(rank, tam, selected_weight, combo);

    bool match = true;
    for (int i = 0; i < n_rows; i++) {
        int parity = 0;
        for (int w = 0; w < selected_weight; w++) {
            int col_idx = d_active_pos[combo[w]];
            parity ^= H[i * m_cols + col_idx];
        }
        if (parity != s[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        if (atomicExch(&d_found_error_idx, (int)tid) == -1) {
            for (int w = 0; w < GRAND_MAX_W; w++) {
                if (w < selected_weight) {
                    d_winning_combo[w] = d_active_pos[combo[w]];
                } else {
                    d_winning_combo[w] = -1;
                }
            }
        }
    }
}

__global__ void search_errors_massively(int *H, int *s, int *d_active_pos, int tam, int n_rows, int m_cols, int max_error_weight) {

    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int weight = -1;
    unsigned long long rank;
    unsigned long long acc = 0;
    

    for (int w = 1; w <= max_error_weight && w <= GRAND_MAX_W && w <= tam; w++) {
        unsigned long long count = d_nCr_table[tam][w];

        if (tid < acc + count) {
            weight = w;
            rank = tid - acc;
            break;
        }

        acc += count;
    }

    if (weight == -1) {
        return;
    }

    int combo[GRAND_MAX_W]; 
    get_combination(rank, tam, weight, combo);

    bool match = true;
    for (int i = 0; i < n_rows; i++) {
        int parity = 0;
        for (int w = 0; w < weight; w++) {
            int col_idx = d_active_pos[combo[w]];
            parity ^= H[i * m_cols + col_idx];
        }
        if (parity != s[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        if (atomicExch(&d_found_error_idx, (int)tid) == -1) {
            for (int w = 0; w < GRAND_MAX_W; w++) {
                if(w < weight){ 
                    d_winning_combo[w] = d_active_pos[combo[w]];
                }
                else {
                    d_winning_combo[w] = -1;
                }
            }
        }
    }
}

__global__ void search_errors_massively_ml(int *H, int *s, int *d_active_pos, int tam, int n_rows, int m_cols, int max_error_weight, float *d_llr) {

    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    int weight = -1;
    unsigned long long rank;
    unsigned long long acc = 0;

    for (int w = 1; w <= max_error_weight && w <= GRAND_MAX_W && w <= tam; w++) {
        unsigned long long count = d_nCr_table[tam][w];

        if (tid < acc + count) {
            weight = w;
            rank = tid - acc;
            break;
        }

        acc += count;
    }

    if (weight == -1) {
        return;
    }

    int combo[GRAND_MAX_W];
    get_combination(rank, tam, weight, combo);

    bool match = true;
    for (int i = 0; i < n_rows; i++) {
        int parity = 0;
        for (int w = 0; w < weight; w++) {
            int col_idx = d_active_pos[combo[w]];
            parity ^= H[i * m_cols + col_idx];
        }
        if (parity != s[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        float cost = 0.0f;
        for (int w = 0; w < weight; w++) {
            cost += d_llr[d_active_pos[combo[w]]];
        }

        unsigned int cost_bits = __float_as_uint(cost);
        unsigned long long key = (((unsigned long long) cost_bits) << 32) | (unsigned int) tid;

        atomicMinULL_ml(&d_best_key_ml, key);
    }
}

// ------------------------------------------------ CPU ----------------------------------------------------------------------------------

static void get_combination_cpu(unsigned long long rank, int n, int k, int *out_combo) {
    int candidate = 0;

    for (int i = 0; i < k; ++i) {
        int next_k = k - 1 - i;
        unsigned long long count = nCr_table[n - 1 - candidate][next_k];

        while (rank >= count && (n - 1 - candidate) > next_k) {
            rank -= count;
            candidate++;
            count = nCr_table[n - 1 - candidate][next_k];
        }

        out_combo[i] = candidate;
        candidate++;
    }
}

static bool check_syndrome_cpu( int *H, int *s, int *positions, int *combo, int weight, int n_rows, int m_cols ) {
    
    for (int r = 0; r < n_rows; r++) {
        int parity = 0;
        for (int w = 0; w < weight; w++) {
            int col_idx = positions[combo[w]];
            parity ^= H[r * m_cols + col_idx];
        }
        if (parity != s[r]) return false;
    }
    return true;
}

static int search_cpu_weight( int *H, int *s, int *positions, int tam, int n_rows, int m_cols, int *e_estimated, int *number_of_iterations, int weight ) {
    
    *number_of_iterations = 0;

    if (weight <= 0 || weight > tam) {
        return 0;
    }

    unsigned long long total = nCr_table[tam][weight];

    for (unsigned long long rank = 0; rank < total; rank++) {
        (*number_of_iterations)++;

        int combo[GRAND_MAX_W];
        get_combination_cpu(rank, tam, weight, combo);

        if (check_syndrome_cpu(H, s, positions, combo, weight, n_rows, m_cols)) {
            for (int i = 0; i < m_cols; i++) {
                e_estimated[i] = 0;
            }

            for (int w = 0; w < weight; w++) {
                e_estimated[positions[combo[w]]] = 1;
            }

            return 1;
        }
    }

    return 0;
}

static int search_cpu_phase(int *H, int *s, int *positions, int tam, int n_rows, int m_cols, int *e_estimated, int *total_iterations, int *phase_weights, int num_phase_weights) {
    
    *total_iterations = 0;

    for (int i = 0; i < num_phase_weights; i++) {
        int weight = phase_weights[i];

        if (weight <= 0 || weight > tam) {
            continue;
        }

        int iterations_this_weight = 0;

        int found = search_cpu_weight(H, s, positions, tam, n_rows, m_cols, e_estimated, &iterations_this_weight, weight );

        *total_iterations += iterations_this_weight;

        if (found) {
            return 1;
        }
    }

    return 0;
}

static void generate_reduce_matrix(int *s, int n, int m, int *H_red, int *H) {
    
    int current_row = 0;

    for (int i = 0; i < n; i++) {
        if (s[i] == 1) {
            for (int j = 0; j < m; j++) {
                H_red[current_row * m + j] = H[i * m + j];
            }
            current_row++;
        }
    }
}

static void sum_columns_H_red(int *H_red, int rows, int cols, int *sum_cols) {
    
    for (int c = 0; c < cols; c++) {
        sum_cols[c] = 0;
        for (int r = 0; r < rows; r++) {
            sum_cols[c] += H_red[r * cols + c];
        }
    }
}

static void compute_column_stats( int *H, int n, int m, int *column_weights, int *max_soma, int *is_regular ) {
    *max_soma = 0;
    *is_regular = 1;

    for (int c = 0; c < m; c++) {
        int col_sum = 0;

        for (int r = 0; r < n; r++) {
            col_sum += H[r * m + c];
        }

        column_weights[c] = col_sum;

        if (col_sum > *max_soma) {
            *max_soma = col_sum;
        }
    }

    if (m > 0) {
        int first_weight = column_weights[0];

        for (int c = 1; c < m; c++) {
            if (column_weights[c] != first_weight) {
                *is_regular = 0;
                break;
            }
        }
    }
}

static void apply_offset_to_sum_cols( int *sum_cols, int *column_weights, int m, int max_soma ) {
    for (int c = 0; c < m; c++) {
        if (column_weights[c] == 0) {
            sum_cols[c] = 0;
            continue;
        }

        int offset = max_soma - column_weights[c];
        sum_cols[c] += offset;
    }
}

static int gf2_row_reduce(int *H, int rows, int cols, int *piv_cols) {
    
    int r = 0;

    for (int c = 0; c < cols && r < rows; c++) {
        int pivot = -1;
        for (int i = r; i < rows; i++) {
            if (H[i * cols + c] & 1) {
                pivot = i;
                break;
            }
        }
        if (pivot == -1) continue;

        if (pivot != r) {
            for (int j = 0; j < cols; j++) {
                int tmp = H[r * cols + j];
                H[r * cols + j] = H[pivot * cols + j];
                H[pivot * cols + j] = tmp;
            }
        }

        for (int i = 0; i < rows; i++) {
            if (i == r) continue;
            if (H[i * cols + c] & 1) {
                for (int j = 0; j < cols; j++) {
                    H[i * cols + j] ^= H[r * cols + j];
                }
            }
        }

        piv_cols[r] = c;
        r++;
    }

    return r;
}

static int in_rowspace(int *H_simplificada, int rows, int cols, int *piv_cols, int rank, int *e) {
    
    for (int r = 0; r < rank; r++) {
        int c = piv_cols[r];
        if (e[c] & 1) {
            for (int j = c; j < cols; j++) {
                e[j] ^= H_simplificada[r * cols + j];
            }
        }
    }

    for (int j = 0; j < cols; j++) {
        if (e[j] & 1) {
            return 0;
        }
    }

    return 1;
}

static void compute_syndrome(int *H, int m, int n, int *e, int *s, int *weight_syndrome){

    * weight_syndrome = 0;
    for (int r = 0; r < m; r++) { 
        s[r] = 0;
        for (int c = 0; c < n; c++) { 
            s[r] ^= (H[r * n + c] & e[c]); 
        }
        if (s[r] == 1) (*weight_syndrome)++;
    }
}

static void calculate_delta(int *e, int *e_est, int m, int *delta) {
    
    for (int i = 0; i < m; i++) {
        delta[i] = e[i] ^ e_est[i];
    }
}

static void extract_positions(int *array_, int cols, int *pos_out, int threshold, int *tam) {
    
    *tam = 0;
    for (int i = 0; i < cols; i++) {
        if (array_[i] >= threshold) {
            pos_out[*tam] = i;
            (*tam)++;
        }
    }
}

static void number_of_threads_for_combinations_prob( int tam, int *phase_weights, int num_phase_weights, unsigned long long *total_threads ) {
    
    *total_threads = 0;
    for (int i = 0; i < num_phase_weights; i++) {
        *total_threads += nCr_cpu(tam, phase_weights[i]);
    }
}

static void number_of_threads_for_combinations(int tam, unsigned long long *total_threads, int max_error_weight) {
    *total_threads = 0;

    if(max_error_weight > GRAND_MAX_W) {
        max_error_weight = GRAND_MAX_W;
    }

    for(int k=1; k<=max_error_weight; k++) {
        *total_threads += nCr_cpu(tam, k);
    }
}

static int equal_vectors(int *a, int *b, int size) {
    
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

static int search_cpu_massively( int *H, int *s, int *positions, int tam, int n_rows, int m_cols, int *e_estimated, int *number_of_iterations, int max_error_weight ) {
    *number_of_iterations = 0;

    if (max_error_weight > GRAND_MAX_W) {
        max_error_weight = GRAND_MAX_W;
    }

    unsigned long long total_threads = 0;

    for (int w = 1; w <= max_error_weight; w++) {
        total_threads += nCr_table[tam][w];
    }

    for (unsigned long long tid = 0; tid < total_threads; tid++) {
        (*number_of_iterations)++;

        int weight = -1;
        unsigned long long rank = 0;
        unsigned long long acc = 0;

        for (int w = 1; w <= max_error_weight; w++) {
            unsigned long long count = nCr_table[tam][w];

            if (tid < acc + count) {
                weight = w;
                rank = tid - acc;
                break;
            }

            acc += count;
        }

        if (weight == -1) {
            continue;
        }

        int combo[GRAND_MAX_W];

        get_combination_cpu(rank, tam, weight, combo);

        if (check_syndrome_cpu(H, s, positions, combo, weight, n_rows, m_cols)) {
            for (int i = 0; i < m_cols; i++) {
                e_estimated[i] = 0;
            }

            for (int w = 0; w < weight; w++) {
                e_estimated[positions[combo[w]]] = 1;
            }

            return 1;
        }
    }

    return 0;
}

static int search_cpu_ml(int *H, int *s, int *positions, int tam, int n_rows, int m_cols, int *e_estimated, int *number_of_iterations, int max_error_weight, float *llr) {

    *number_of_iterations = 0;

    if (max_error_weight > GRAND_MAX_W) {
        max_error_weight = GRAND_MAX_W;
    }

    int sorted_pos[INTERNAL_MAX_N];
    for (int i = 0; i < tam; i++) sorted_pos[i] = positions[i];

    for (int i = 1; i < tam; i++) {
        int key = sorted_pos[i];
        float key_llr = llr[key];
        int j = i - 1;
        while (j >= 0 && llr[sorted_pos[j]] > key_llr) {
            sorted_pos[j + 1] = sorted_pos[j];
            j--;
        }
        sorted_pos[j + 1] = key;
    }

    int max_w = (max_error_weight < tam) ? max_error_weight : tam;

    float prefix[GRAND_MAX_W + 1];
    prefix[0] = 0.0f;
    for (int w = 1; w <= max_w; w++) {
        prefix[w] = prefix[w - 1] + llr[sorted_pos[w - 1]];
    }

    bool best_found = false;
    float best_cost = 0.0f;
    int best_weight = -1;
    int best_combo[GRAND_MAX_W];

    for (int weight = 1; weight <= max_w; weight++) {

        if (best_found && prefix[weight] >= best_cost) {
            break;
        }

        unsigned long long count_w = nCr_table[tam][weight];

        for (unsigned long long rank = 0; rank < count_w; rank++) {
            (*number_of_iterations)++;

            int combo[GRAND_MAX_W];
            get_combination_cpu(rank, tam, weight, combo);

            if (check_syndrome_cpu(H, s, sorted_pos, combo, weight, n_rows, m_cols)) {

                float cost = 0.0f;
                for (int w = 0; w < weight; w++) {
                    cost += llr[sorted_pos[combo[w]]];
                }

                if (!best_found || cost < best_cost) {
                    best_found = true;
                    best_cost = cost;
                    best_weight = weight;
                    for (int w = 0; w < GRAND_MAX_W; w++) {
                        best_combo[w] = (w < weight) ? sorted_pos[combo[w]] : -1;
                    }
                }
            }
        }
    }

    if (best_found) {
        for (int i = 0; i < m_cols; i++) e_estimated[i] = 0;
        for (int w = 0; w < GRAND_MAX_W; w++) {
            if (best_combo[w] != -1) e_estimated[best_combo[w]] = 1;
        }
        return 1;
    }
    return 0;
}

static void decode_winning_combo_ml(unsigned long long tid, int tam, int max_error_weight, int *active_pos, int *out_weight, int out_combo[GRAND_MAX_W]) {

    if (max_error_weight > GRAND_MAX_W) max_error_weight = GRAND_MAX_W;

    unsigned long long acc = 0;
    int weight = -1;
    unsigned long long rank = 0;

    for (int w = 1; w <= max_error_weight && w <= tam; w++) {
        unsigned long long count = nCr_table[tam][w];
        if (tid < acc + count) {
            weight = w;
            rank = tid - acc;
            break;
        }
        acc += count;
    }

    int local_combo[GRAND_MAX_W];
    get_combination_cpu(rank, tam, weight, local_combo);

    *out_weight = weight;
    for (int w = 0; w < GRAND_MAX_W; w++) {
        out_combo[w] = (w < weight) ? active_pos[local_combo[w]] : -1;
    }
}

static int read_llr_from_file_ml(const char *llr_file, float *llr, int m) {

    FILE *file = fopen(llr_file, "r");
    if (!file) {
        fprintf(stderr, "Error opening LLR file: %s\n", llr_file);
        return 0;
    }

    for (int i = 0; i < m; i++) {
        if (fscanf(file, "%f", &llr[i]) != 1) {
            fprintf(stderr, "Error reading LLR file (expected %d values): %s\n", m, llr_file);
            fclose(file);
            return 0;
        }
    }

    fclose(file);
    return 1;
}

void print_array_matrix(int *a, const char *name, int rows, int cols){
    printf("%s = \n ", name);
    for(int i=0; i<rows; i++){
        printf("[ ");
        for(int j=0; j<cols; j++){
            printf("%d ", a[i * cols + j]);
        }
        printf("]\n ");
    }
}

// ------------------------------------------------ PROBABILIDADES COM MÁSCARAS ----------------------------------------------------------------------------------

static int read_phase_orders_from_file( const char *probability_file, PhaseOrderMask *masks, int max_masks ) {
    
    FILE *file = fopen(probability_file, "r");
    if (!file) {
        fprintf(stderr, "Error opening probability file: %s\n", probability_file);
        return 0;
    }

    int count = 0;

    while (count < max_masks) {
        PhaseOrderMask temp;

        int read = fscanf( file, "%d %d %d %d %f %f %f", &temp.tam_upper, &temp.phase_order[0], &temp.phase_order[1], &temp.phase_order[2], &temp.phase_prob[0], &temp.phase_prob[1], &temp.phase_prob[2]);

        if (read != 7) {
            break;
        }

        masks[count++] = temp;
    }

    fclose(file);
    return count;
}

static PhaseOrderMask *get_mask_for_tam(int tam, PhaseOrderMask *masks, int n_masks) {
    
    for (int i = 0; i < n_masks; i++) {
        if (tam <= masks[i].tam_upper) {
            return &masks[i];
        }
    }
    return &masks[n_masks - 1];
}

static void get_weights_for_phase_id(int phase_id, int tam, int *phase_weights, int *len) {
    
    *len = 0;

    switch (phase_id) {
        case 0:
            for (int w = 1; w <= 5 && w <= tam && w <= GRAND_MAX_W; w++) {
                phase_weights[(*len)++] = w;
            }
            break;

        case 1:
            if (6 <= tam && 6 <= GRAND_MAX_W) phase_weights[(*len)++] = 6;
            break;

        case 2:
            if (7 <= tam && 7 <= GRAND_MAX_W) phase_weights[(*len)++] = 7;
            break;

        default:
            break;
    }
}

// ------------------------------------------------ API PUBLICA ----------------------------------------------------------------------------------

extern "C" int grand_decode_prob_cc(const int *Hx, const int *Hz, int n, int m, int *errors, int *syndromes, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results, char type_of_error, const char *probability_file) {
    
    setup_nCr_table_internal();

    int *H = (int *) malloc(n * m * sizeof(int));
    int *H_simplificada = (int *) malloc(n * m * sizeof(int));

    int *e_pos = (int *) malloc(m * sizeof(int));
    int *e_est_pos = (int *) malloc(m * sizeof(int));
    int *e_est = (int *) malloc(m * sizeof(int));

    int *syndrome = (int *) malloc(n * sizeof(int));
    int *s_est = (int *) malloc(n * sizeof(int));
    int *s_pos = (int *) malloc(n * sizeof(int));
    int *s_est_pos = (int *) malloc(n * sizeof(int));

    int *sum_cols = (int *) malloc(m * sizeof(int));
    int *h_active_pos = (int *) malloc(m * sizeof(int));

    int *column_weights = (int *) malloc(m * sizeof(int));

    int *delta = (int *) malloc(m * sizeof(int));
    int *piv_cols = (int *) malloc(n * sizeof(int));
    int *win_combo = (int *) malloc(GRAND_MAX_W * sizeof(int));

    int *d_s, *d_sum_cols, *d_active_pos, *d_H, *d_active_rows, *d_phase_weights;


    cudaEvent_t t_sum_1, t_sum_2, t_search_1, t_search_2;

    PhaseOrderMask masks[GRAND_MAX_MASKS];

    int num_masks = 0, rank = 0,  reset_val = -1, h_found_idx = -1, found = 0, found_weight = 0, degenerate = 0, is_equal = 0, max_soma = 0, is_regular = 0;
    int iteration = 0, tam = 0, weight_syndrome = 0, hibrido = -2, original_weight = 0, shot = 0, num_phases, using_syndromes=0;
    unsigned long long total_tested_combinations = 0;

    double time_sum_ms = 0.0, total_time_search_ms = 0.0, transfer_time_ms = 0.0, total_to_find_ms = 0.0;
    float time_sum_ms_f = 0.0f;

    if (errors == NULL){
        using_syndromes = 1;
    }

    if (type_of_error == 'x' || type_of_error == 'X') {
        memcpy(H_simplificada, Hx, n * m * sizeof(int));
        memcpy(H, Hz, n * m * sizeof(int));
    } else if (type_of_error == 'z' || type_of_error == 'Z') {
        memcpy(H_simplificada, Hz, n * m * sizeof(int));
        memcpy(H, Hx, n * m * sizeof(int));
    }
    else{
        free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(s_pos); free(delta); free(win_combo); free(piv_cols);
        free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
        free(column_weights);
        fprintf(stderr, "Invalid type_of_error: %c. Must be 'x' or 'z'.\n", type_of_error);
        return -1;
    }

    compute_column_stats(H, n, m, column_weights, &max_soma, &is_regular);
    num_masks = read_phase_orders_from_file(probability_file, masks, GRAND_MAX_MASKS);
    rank = gf2_row_reduce(H_simplificada, n, m, piv_cols);

    if (config->max_error_weight == 6) {
        num_phases = 2;
    }
    else if (config->max_error_weight == 7) {
        num_phases = 3;
    }
    else if (config->max_error_weight < 6 && config->max_error_weight > 0) {
        num_phases = 1;
    }

    else{
        free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(s_pos); free(delta); free(win_combo); free(piv_cols);
        free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
        free(column_weights);
        fprintf(stderr, "Invalid max_error_weight: %d. Must be 6 or 7.\n", config->max_error_weight);
        return -1;
    }

    cudaMalloc(&d_s, n * sizeof(int));
    cudaMalloc(&d_sum_cols, m * sizeof(int));
    cudaMalloc(&d_active_pos, m * sizeof(int));
    cudaMalloc(&d_H, n * m * sizeof(int));
    cudaMalloc(&d_active_rows, n * sizeof(int));
    cudaMalloc(&d_phase_weights, GRAND_MAX_W * sizeof(int));

    cudaEventCreate(&t_sum_1);
    cudaEventCreate(&t_sum_2);
    cudaEventCreate(&t_search_1);
    cudaEventCreate(&t_search_2);

    cudaMemcpy(d_H, H, n * m * sizeof(int), cudaMemcpyHostToDevice);


    for (shot=0; shot < num_shots; shot++){

        GrandDecoderResult *result = &results[shot];

        memset(e_est, 0, m * sizeof(int)); memset(syndrome, 0, n * sizeof(int)); memset(result, 0, sizeof(*result)); memset(e_pos, 0, m * sizeof(int)); memset(e_est_pos, 0, m * sizeof(int)); memset(s_est, 0, n * sizeof(int));
        found = 0, found_weight = 0, degenerate = 0, is_equal = 0, iteration = 0, tam = 0, weight_syndrome = 0, hibrido = -2, original_weight = 0;
        total_tested_combinations = 0;
        time_sum_ms = 0.0, total_time_search_ms = 0.0, transfer_time_ms = 0.0, total_to_find_ms = 0.0;
        time_sum_ms_f = 0.0f;


        int *e_real = NULL;

        if (using_syndromes == 0) {
            e_real = &errors[shot * m];
            extract_positions(e_real, m, e_pos, 1, &original_weight);
            compute_syndrome( H, n, m, e_real, syndrome, &weight_syndrome );
        }
        else {
            memcpy( syndrome, &syndromes[shot * n], n * sizeof(int) );
        }

        extract_positions(syndrome, n, s_pos, 1, &weight_syndrome);

        if (weight_syndrome == 0){
            result->found = 0; result->found_weight = 0; result->degenerate = 0; result->is_equal = 0; result->iteration = 0, result->original_weight = original_weight;
            result->tam = 0; result->total_tested_combinations = 0; result->hibrido = -2; result->time_sum_ms = 0.0;
            result->total_time_search_ms = 0.0; result->transfer_time_ms = 0.0; result->total_to_find_ms = 0.0;

            if (using_syndromes == 0) {
                printf("shot %d: found=%d, original_weight=%d, found_weight=%d, degenerate=%d, is_equal=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
                shot + 1, result->found, result->original_weight, result->found_weight, result->degenerate, result->is_equal, result->iteration,
                result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);
            }

            else {
                printf("shot %d: found=%d, found_weight=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
                shot + 1, result->found, result->found_weight, result->iteration,result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);
            }

            continue;
        }

        cudaMemcpy(d_s, syndrome, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_active_rows, s_pos, weight_syndrome * sizeof(int), cudaMemcpyHostToDevice);
        
        if (config->use_gpu_sum) {
            cudaEventRecord(t_sum_1);
            int blocks_sum = (m + config->threads_per_block - 1) / config->threads_per_block;
            sum_columns_H_gpu<<<blocks_sum, config->threads_per_block>>>(d_H, d_active_rows, weight_syndrome, d_sum_cols, m);
            cudaEventRecord(t_sum_2);
            cudaEventSynchronize(t_sum_2);
            cudaEventElapsedTime((float*)&time_sum_ms_f, t_sum_1, t_sum_2);
            time_sum_ms = (double)time_sum_ms_f;

            cudaMemcpy(sum_cols, d_sum_cols, m * sizeof(int), cudaMemcpyDeviceToHost);

        } else if (!config->use_gpu_sum) {
            int *H_red = (int*)malloc(weight_syndrome * m * sizeof(int));

            auto t1 = std::chrono::steady_clock::now();
            generate_reduce_matrix(syndrome, n, m, H_red, H);
            sum_columns_H_red(H_red, weight_syndrome, m, sum_cols);
            auto t2 = std::chrono::steady_clock::now();

            time_sum_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            free(H_red);
        }

        else{
            fprintf(stderr, "Invalid value for use_gpu_sum: %d\n", config->use_gpu_sum);
            free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(s_pos); free(delta); free(win_combo); free(piv_cols);
            free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
            free(column_weights);
            return -1;
        }

        if (!is_regular) {
            apply_offset_to_sum_cols(sum_cols, column_weights, m, max_soma);
        }

        auto t_total_1 = std::chrono::steady_clock::now();
        while (iteration < max_soma && iteration < 3) {
            int threshold = max_soma - iteration;
            iteration++;

            h_found_idx = -1; found = 0;

            extract_positions(sum_cols, m, h_active_pos, threshold, &tam);

            if (tam <= 0) {
                continue;
            }

            PhaseOrderMask *pm = get_mask_for_tam(tam, masks, num_masks);

            if ((tam > config->threshold_for_hybrid && tam <= config->max_size_active_positions && config->mode_of_search == 1) || (config->mode_of_search == 0)) {
                if (hibrido == -1) {
                    hibrido = 1;
                } else {
                    hibrido = 0;
                }

                auto ttr1 = std::chrono::steady_clock::now();
                cudaMemcpyToSymbol(d_found_error_idx, &reset_val, sizeof(int));
                cudaMemcpy(d_active_pos, h_active_pos, tam * sizeof(int), cudaMemcpyHostToDevice);
                auto ttr2 = std::chrono::steady_clock::now();
                transfer_time_ms += std::chrono::duration<double, std::milli>(ttr2 - ttr1).count();

                for (int k = 0; k < num_phases; k++) {
                    int phase_id = pm->phase_order[k];
                    int h_phase_weights[GRAND_MAX_W];
                    int len_phase = 0;
                    unsigned long long phase_threads = 0;

                    get_weights_for_phase_id(phase_id, tam, h_phase_weights, &len_phase);
                    if (len_phase == 0) continue;

                    number_of_threads_for_combinations_prob(tam, h_phase_weights, len_phase, &phase_threads);
                    if (phase_threads == 0) continue;

                    total_tested_combinations += phase_threads;

                    auto ttr3 = std::chrono::steady_clock::now();
                    cudaMemcpy(d_phase_weights, h_phase_weights, len_phase * sizeof(int), cudaMemcpyHostToDevice);
                    auto ttr4 = std::chrono::steady_clock::now();
                    transfer_time_ms += std::chrono::duration<double, std::milli>(ttr4 - ttr3).count();

                    unsigned int blocks_search = (unsigned int)((phase_threads + config->threads_per_block - 1) / config->threads_per_block);

                    cudaEventRecord(t_search_1);
                    search_errors_massively_prob<<<blocks_search, config->threads_per_block>>>( d_H, d_s, d_active_pos, tam, n, m, d_phase_weights, len_phase );
                    cudaEventRecord(t_search_2);
                    cudaEventSynchronize(t_search_2);

                    float phase_search_ms = 0.0f;
                    cudaEventElapsedTime(&phase_search_ms, t_search_1, t_search_2);
                    total_time_search_ms += phase_search_ms;

                    auto ttr5 = std::chrono::steady_clock::now();
                    cudaMemcpyFromSymbol(&h_found_idx, d_found_error_idx, sizeof(int));
                    auto ttr6 = std::chrono::steady_clock::now();
                    transfer_time_ms += std::chrono::duration<double, std::milli>(ttr6 - ttr5).count();

                    if (h_found_idx != -1) {
                        break;
                    }
                }
            } else if ((tam <= config->threshold_for_hybrid && config->mode_of_search == 1) || (config->mode_of_search == -1)) {
                hibrido = -1;

                auto tc1 = std::chrono::steady_clock::now();

                for (int k = 0; k < num_phases; k++) {
                    int phase_id = pm->phase_order[k];
                    int h_phase_weights[GRAND_MAX_W];
                    int len_phase = 0;
                    int iterations_phase = 0;

                    get_weights_for_phase_id(phase_id, tam, h_phase_weights, &len_phase);
                    if (len_phase == 0) continue;

                    found = search_cpu_phase(H, syndrome, h_active_pos, tam, n, m, e_est, &iterations_phase, h_phase_weights, len_phase);

                    total_tested_combinations += (unsigned long long)iterations_phase;

                    if (found) {
                        break;
                    }
                }

                auto tc2 = std::chrono::steady_clock::now();
                total_time_search_ms += std::chrono::duration<double, std::milli>(tc2 - tc1).count();
            }

            if (h_found_idx != -1 || found) {
                auto t_total_2 = std::chrono::steady_clock::now();
                total_to_find_ms = std::chrono::duration<double, std::milli>(t_total_2 - t_total_1).count();
                break;
            }
        }

        if (h_found_idx != -1) {
            found = 1;

            cudaMemcpyFromSymbol(win_combo, d_winning_combo, GRAND_MAX_W * sizeof(int));

            for (int w = 0; w < GRAND_MAX_W; w++) {
                if (win_combo[w] != -1) {
                    found_weight++;
                    e_est[win_combo[w]] = 1;
                }
            }
        }

        if (found) {
            extract_positions(e_est, m, e_est_pos, 1, &found_weight);
            if (using_syndromes == 0) {
                is_equal = equal_vectors(e_real, e_est, m);
                calculate_delta(e_real, e_est, m, delta);
                degenerate = in_rowspace(H_simplificada, n, m, piv_cols, rank, delta);
            }
        }

        result->found = found; result->found_weight = found_weight; result->degenerate = degenerate; result->is_equal = is_equal; result->iteration = iteration;
        result->tam = tam; result->total_tested_combinations = total_tested_combinations; result->hibrido = hibrido; result->time_sum_ms = time_sum_ms;
        result->total_time_search_ms = total_time_search_ms; result->transfer_time_ms = transfer_time_ms; result->total_to_find_ms = total_to_find_ms;
        result->original_weight = original_weight;

        if(using_syndromes==0){
            printf("shot %d: found=%d, original_weight=%d, found_weight=%d, degenerate=%d, is_equal=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
            shot + 1, result->found, result->original_weight, result->found_weight, result->degenerate, result->is_equal, result->iteration,
            result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);
        }
        else{
            printf("shot %d: found=%d, found_weight=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
            shot + 1, result->found, result->found_weight, result->iteration,result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);
        }
    }

    cudaEventDestroy(t_sum_1);
    cudaEventDestroy(t_sum_2);
    cudaEventDestroy(t_search_1);
    cudaEventDestroy(t_search_2);

    cudaFree(d_s); cudaFree(d_sum_cols); cudaFree(d_active_pos);cudaFree(d_H); cudaFree(d_active_rows); cudaFree(d_phase_weights);

    free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(s_pos); free(delta); free(win_combo); free(piv_cols);

    return 0;
} 

extern "C" int grand_decode_cc(const int *Hx, const int *Hz, int n, int m, int *errors, int *syndromes, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results, char type_of_error) {

        setup_nCr_table_internal();

        int *H = (int *) malloc(n * m * sizeof(int));
        int *H_simplificada = (int *) malloc(n * m * sizeof(int));

        int *e_pos = (int *) malloc(m * sizeof(int));
        int *e_est_pos = (int *) malloc(m * sizeof(int));
        int *e_est = (int *) malloc(m * sizeof(int));

        int *syndrome = (int *) malloc(n * sizeof(int));
        int *s_pos = (int *) malloc(n * sizeof(int));
        int *s_est = (int *) malloc(n * sizeof(int));
        int *s_est_pos = (int *) malloc(n * sizeof(int));

        int *sum_cols = (int *) malloc(m * sizeof(int));
        int *h_active_pos = (int *) malloc(m * sizeof(int));

        int *delta = (int *) malloc(m * sizeof(int));
        int *piv_cols = (int *) malloc(n * sizeof(int));
        int *win_combo = (int *) malloc(GRAND_MAX_W * sizeof(int));

        int *column_weights = (int *) malloc(m * sizeof(int));

        int *d_s, *d_sum_cols, *d_active_pos, *d_H, *d_active_rows;

        cudaEvent_t t_sum_1, t_sum_2, t_search_gpu_1, t_search_gpu_2;

        int rank = 0,  reset_val = -1, h_found_idx = -1, found = 0, found_weight = 0, degenerate = 0, is_equal = 0, max_soma = 0, is_regular = 0;
        int iteration = 0, tam = 0, weight_syndrome = 0, hibrido = -2, number_of_iterations = 0, shot = 0, original_weight = 0, using_syndromes=0;
        unsigned long long total_tested_combinations = 0, total_threads = 0;
        unsigned int blocks_search;

        double time_sum_ms = 0.0, total_time_search_ms = 0.0, transfer_time_ms = 0.0, total_to_find_ms = 0.0, time_search_ms = 0.0;
        float time_sum_ms_f = 0.0f, time_search_ms_f = 0.0f;


        if (errors == NULL){
            using_syndromes = 1;
        }

        if (type_of_error == 'x' || type_of_error == 'X') {
            memcpy(H_simplificada, Hx, n * m * sizeof(int));
            memcpy(H, Hz, n * m * sizeof(int));
        } else if (type_of_error == 'z' || type_of_error == 'Z') {
            memcpy(H_simplificada, Hz, n * m * sizeof(int));
            memcpy(H, Hx, n * m * sizeof(int));
        }
        else{
            free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(s_pos); free(delta); free(win_combo); free(piv_cols);
            free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
            free(column_weights);
            fprintf(stderr, "Invalid type_of_error: %c. Must be 'x' or 'z'.\n", type_of_error);
            return -1;
        }

        if (config->max_error_weight <= 0 || config->max_error_weight > 7) {
            free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(s_pos); free(delta); free(win_combo); free(piv_cols);
            free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
            free(column_weights);
            fprintf(stderr, "Invalid max_error_weight: %d. Must be between 1 and 7.\n", config->max_error_weight);
            return -1;
        }

        compute_column_stats(H, n, m, column_weights, &max_soma, &is_regular);
        rank = gf2_row_reduce(H_simplificada, n, m, piv_cols);

        cudaMalloc(&d_s, n * sizeof(int));
        cudaMalloc(&d_sum_cols, m * sizeof(int));
        cudaMalloc(&d_active_pos, m * sizeof(int));
        cudaMalloc(&d_H, n * m * sizeof(int));
        cudaMalloc(&d_active_rows, n * sizeof(int));

        cudaEventCreate(&t_sum_1);
        cudaEventCreate(&t_sum_2);
        cudaEventCreate(&t_search_gpu_1);
        cudaEventCreate(&t_search_gpu_2);

        cudaMemcpy(d_H, H, n * m * sizeof(int), cudaMemcpyHostToDevice);


        for (shot = 0; shot < num_shots; shot++) {
    
            GrandDecoderResult *result = &results[shot];

            memset(e_est, 0, m * sizeof(int)); memset(syndrome, 0, n * sizeof(int)); memset(result, 0, sizeof(*result)); memset(e_pos, 0, m * sizeof(int)); memset(e_est_pos, 0, m * sizeof(int)); memset(s_est, 0, n * sizeof(int));
            found = 0, found_weight = 0, degenerate = 0, is_equal = 0, iteration = 0, tam = 0, weight_syndrome = 0, hibrido = -2, number_of_iterations = 0, original_weight = 0;
            total_tested_combinations = 0, total_threads = 0, blocks_search = 0;
            time_sum_ms = 0.0, total_time_search_ms = 0.0, transfer_time_ms = 0.0, total_to_find_ms = 0.0, time_search_ms = 0.0;
            time_sum_ms_f = 0.0f, time_search_ms_f = 0.0f;

            int *e_real = NULL;

            if (using_syndromes == 0) {
                e_real = &errors[shot * m];
                extract_positions(e_real, m, e_pos, 1, &original_weight);
                compute_syndrome( H, n, m, e_real, syndrome, &weight_syndrome );
            }
            else {
                memcpy( syndrome, &syndromes[shot * n], n * sizeof(int) );
            }

            extract_positions(syndrome, n, s_pos, 1, &weight_syndrome);
            

            if (weight_syndrome == 0) {

                result->found = 0; result->found_weight = 0; result->degenerate = 0; result->is_equal = 0; result->iteration = 0, result->original_weight = original_weight;
                result->tam = 0; result->total_tested_combinations = 0; result->hibrido = -2; result->time_sum_ms = 0.0;
                result->total_time_search_ms = 0.0; result->transfer_time_ms = 0.0; result->total_to_find_ms = 0.0;

                if (using_syndromes == 0) {
                    printf("shot %d: found=%d, original_weight=%d, found_weight=%d, degenerate=%d, is_equal=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
                    shot + 1, result->found, result->original_weight, result->found_weight, result->degenerate, result->is_equal, result->iteration,
                    result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);
                }

                else {
                    printf("shot %d: found=%d, found_weight=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
                    shot + 1, result->found, result->found_weight, result->iteration,result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);
                }

                continue;
            }

            cudaMemcpy(d_s, syndrome, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_active_rows, s_pos, weight_syndrome * sizeof(int), cudaMemcpyHostToDevice);



            if (config->use_gpu_sum) {
                cudaEventRecord(t_sum_1);
                int blocks_sum = (m + config->threads_per_block - 1) / config->threads_per_block;
                sum_columns_H_gpu<<<blocks_sum, config->threads_per_block>>>(d_H, d_active_rows, weight_syndrome, d_sum_cols, m);
                cudaEventRecord(t_sum_2);
                cudaEventSynchronize(t_sum_2);
                cudaEventElapsedTime((float*)&time_sum_ms_f, t_sum_1, t_sum_2);
                time_sum_ms = (double)time_sum_ms_f;

                cudaMemcpy(sum_cols, d_sum_cols, m * sizeof(int), cudaMemcpyDeviceToHost);

            } else if (!config->use_gpu_sum) {
                int *H_red = (int*)malloc(weight_syndrome * m * sizeof(int));

                auto t1 = std::chrono::steady_clock::now();
                generate_reduce_matrix(syndrome, n, m, H_red, H);
                sum_columns_H_red(H_red, weight_syndrome, m, sum_cols);
                auto t2 = std::chrono::steady_clock::now();

                time_sum_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                free(H_red);
            }

            else{
                fprintf(stderr, "Invalid value for use_gpu_sum: %d\n", config->use_gpu_sum);
                free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(s_pos); free(delta); free(win_combo); free(piv_cols);
                free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
                free(column_weights);
                return -1;
            }

            if (!is_regular) {
                apply_offset_to_sum_cols(sum_cols, column_weights, m, max_soma);
            }

            auto t_total_1 = std::chrono::steady_clock::now();
            while (iteration < max_soma && iteration < 3) {
                int threshold = max_soma - iteration;
                iteration++;

                h_found_idx = -1; found = 0;

                extract_positions(sum_cols, m, h_active_pos, threshold, &tam);

                if (tam <= 0) {
                    continue;
                }

                if ((tam > config->threshold_for_hybrid && tam <= config->max_size_active_positions && config->mode_of_search == 1) || (config->mode_of_search == 0)) {
                    auto t_transfer_1 = std::chrono::steady_clock::now();
                    cudaMemcpyToSymbol(d_found_error_idx, &reset_val, sizeof(int));
                    cudaMemcpy(d_active_pos, h_active_pos, tam * sizeof(int), cudaMemcpyHostToDevice);
                    auto t_transfer_2 = std::chrono::steady_clock::now();
                    transfer_time_ms += std::chrono::duration<double, std::milli>(t_transfer_2 - t_transfer_1).count();
                    
                    number_of_threads_for_combinations(tam, &total_threads, config->max_error_weight);
                    total_tested_combinations += total_threads;

                    blocks_search = (total_threads + config->threads_per_block - 1) / config->threads_per_block;

                    cudaEventRecord(t_search_gpu_1);
                    search_errors_massively<<<blocks_search, config->threads_per_block>>>(d_H, d_s, d_active_pos, tam, n, m, config->max_error_weight);
                    cudaEventRecord(t_search_gpu_2);
                    cudaEventSynchronize(t_search_gpu_2);
                    cudaEventElapsedTime(&time_search_ms_f, t_search_gpu_1, t_search_gpu_2);
                    time_search_ms = (double)time_search_ms_f;

                    auto t_transfer_3 = std::chrono::steady_clock::now();
                    cudaMemcpyFromSymbol(&h_found_idx, d_found_error_idx, sizeof(int));
                    auto t_transfer_4 = std::chrono::steady_clock::now();
                    transfer_time_ms += std::chrono::duration<double, std::milli>(t_transfer_4 - t_transfer_3).count();
                    if(hibrido==-1){
                        hibrido = 1;
                    }
                    else{
                        hibrido = 0;
                    }
                }

                else if ((tam <= config->threshold_for_hybrid && config->mode_of_search == 1) || (config->mode_of_search == -1)) {
                    hibrido = -1;
                    auto t_search_cpu_1 = std::chrono::steady_clock::now();
                    found = search_cpu_massively(H, syndrome, h_active_pos, tam, n, m, e_est, &number_of_iterations, config->max_error_weight);
                    auto t_search_cpu_2 = std::chrono::steady_clock::now();
                    time_search_ms = std::chrono::duration<double, std::milli>(t_search_cpu_2 - t_search_cpu_1).count();
                    total_tested_combinations += number_of_iterations;
                }

                total_time_search_ms += time_search_ms;

                if (h_found_idx != -1 || found) {
                    auto t_total_2 = std::chrono::steady_clock::now();
                    total_to_find_ms = std::chrono::duration<double, std::milli>(t_total_2 - t_total_1).count();
                    break;
                }
            }


            if (h_found_idx != -1) {
                found = 1;

                cudaMemcpyFromSymbol(win_combo, d_winning_combo, GRAND_MAX_W * sizeof(int));

                for (int w = 0; w < GRAND_MAX_W; w++) {
                    if (win_combo[w] != -1) {
                        found_weight++;
                        e_est[win_combo[w]] = 1;
                    }
                }
            }

            if (found) {
                extract_positions(e_est, m, e_est_pos, 1, &found_weight);
                if (using_syndromes == 0) {
                    is_equal = equal_vectors(e_real, e_est, m);
                    calculate_delta(e_real, e_est, m, delta);
                    degenerate = in_rowspace(H_simplificada, n, m, piv_cols, rank, delta);
                }
            }

        result->found = found; result->found_weight = found_weight; result->degenerate = degenerate; result->is_equal = is_equal; result->iteration = iteration;
        result->tam = tam; result->total_tested_combinations = total_tested_combinations; result->hibrido = hibrido; result->time_sum_ms = time_sum_ms;
        result->total_time_search_ms = total_time_search_ms; result->transfer_time_ms = transfer_time_ms; result->total_to_find_ms = total_to_find_ms; result->original_weight = original_weight;
    
        if(using_syndromes==0){
            printf("shot %d: found=%d, original_weight=%d, found_weight=%d, degenerate=%d, is_equal=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
            shot + 1, result->found, result->original_weight, result->found_weight, result->degenerate, result->is_equal, result->iteration,
            result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);
        }
        else{
            printf("shot %d: found=%d, found_weight=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
            shot + 1, result->found, result->found_weight, result->iteration,result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);
        }
    }


        cudaEventDestroy(t_sum_1);
        cudaEventDestroy(t_sum_2);
        cudaEventDestroy(t_search_gpu_1);
        cudaEventDestroy(t_search_gpu_2);

        cudaFree(d_s); cudaFree(d_sum_cols); cudaFree(d_active_pos);cudaFree(d_H); cudaFree(d_active_rows); 

        free(H); free(syndrome); free(H_simplificada); free(sum_cols); free(h_active_pos); free(s_pos); free(delta); 
        free(win_combo); free(piv_cols); free(e_est_pos); free(s_est); free(s_est_pos); free(e_pos); free(e_est); free(column_weights);

        return 0;

    }


extern "C" int grand_decode_cln(int *H_dem, const int *L_dem, int n, int m, int n_logicos, int *detectors, int *logicals, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results) {

        setup_nCr_table_internal();

        int *H = (int *) malloc(n * m * sizeof(int));
        memcpy(H, H_dem, n * m * sizeof(int)); 
        int *H_simplificada = (int *) malloc(n * m * sizeof(int));

        int *e_pos = (int *) malloc(m * sizeof(int));
        int *e_est_pos = (int *) malloc(m * sizeof(int));
        int *e_est = (int *) malloc(m * sizeof(int));

        int *syndrome = (int *) malloc(n * sizeof(int));
        int *detector_pos = (int *) malloc(n * sizeof(int));
        int *s_est = (int *) malloc(n * sizeof(int));
        int *s_est_pos = (int *) malloc(n * sizeof(int));

        int *sum_cols = (int *) malloc(m * sizeof(int));
        int *h_active_pos = (int *) malloc(m * sizeof(int));

        int *delta = (int *) malloc(m * sizeof(int));
        int *piv_cols = (int *) malloc(n * sizeof(int));
        int *win_combo = (int *) malloc(GRAND_MAX_W * sizeof(int));

        int *logical_est = (int *) calloc(n_logicos, sizeof(int));
        int *logicals_in_error = (int *) calloc(n_logicos, sizeof(int));

        int *column_weights = (int *) malloc(m * sizeof(int));

        int *d_s, *d_sum_cols, *d_active_pos, *d_H, *d_active_rows;

        cudaEvent_t t_sum_1, t_sum_2, t_search_gpu_1, t_search_gpu_2;

        int reset_val = -1, h_found_idx = -1, found = 0, found_weight = 0, degenerate = 0, max_soma = 0, is_regular = 0;
        int iteration = 0, tam = 0, tam_old = 0, weight_syndrome = 0, hibrido = -2, number_of_iterations = 0, shot = 0;
        unsigned long long total_tested_combinations = 0, total_threads = 0;
        unsigned int blocks_search;

        double time_sum_ms = 0.0, total_time_search_ms = 0.0, transfer_time_ms = 0.0, total_to_find_ms = 0.0, time_search_ms = 0.0;
        float time_sum_ms_f = 0.0f, time_search_ms_f = 0.0f;



        if (config->max_error_weight <= 0 || config->max_error_weight > 7) {
            free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(delta); free(win_combo); free(piv_cols);
            free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
            free(column_weights); free(logical_est); free(logicals_in_error);
            fprintf(stderr, "Invalid max_error_weight: %d. Must be between 1 and 7.\n", config->max_error_weight);
            return -1;
        }

        compute_column_stats(H_dem, n, m, column_weights, &max_soma, &is_regular);

        cudaMalloc(&d_s, n * sizeof(int));
        cudaMalloc(&d_sum_cols, m * sizeof(int));
        cudaMalloc(&d_active_pos, m * sizeof(int));
        cudaMalloc(&d_H, n * m * sizeof(int));
        cudaMalloc(&d_active_rows, n * sizeof(int));

        cudaEventCreate(&t_sum_1);
        cudaEventCreate(&t_sum_2);
        cudaEventCreate(&t_search_gpu_1);
        cudaEventCreate(&t_search_gpu_2);

        cudaMemcpy(d_H, H, n * m * sizeof(int), cudaMemcpyHostToDevice);


        for (shot = 0; shot < num_shots; shot++) {
    
            GrandDecoderResult *result = &results[shot];

            memset(e_est, 0, m * sizeof(int)); memset(syndrome, 0, n * sizeof(int)); memset(result, 0, sizeof(*result)); memset(e_pos, 0, m * sizeof(int)); memset(e_est_pos, 0, m * sizeof(int)); memset(s_est, 0, n * sizeof(int));
            memset(logical_est, 0, n_logicos * sizeof(int)); memset(logicals_in_error, 0, n_logicos * sizeof(int)); 
            found = 0, found_weight = 0, degenerate = 0, iteration = 0, tam = 0, tam_old = 0, weight_syndrome = 0, hibrido = -2, number_of_iterations = 0;
            total_tested_combinations = 0, total_threads = 0, blocks_search = 0;
            time_sum_ms = 0.0, total_time_search_ms = 0.0, transfer_time_ms = 0.0, total_to_find_ms = 0.0, time_search_ms = 0.0;
            time_sum_ms_f = 0.0f, time_search_ms_f = 0.0f;

            int *detector = NULL;
            int *logical = NULL;

            detector = &detectors[shot * n];
            logical = &logicals[shot * n_logicos];
            
            memcpy(syndrome, detector, n * sizeof(int));  
            extract_positions(detector, n, detector_pos, 1, &weight_syndrome);
            

            if (weight_syndrome == 0) {

                result->found = 0; result->found_weight = 0; result->degenerate = 0; result->iteration = 0;
                result->tam = 0; result->total_tested_combinations = 0; result->hibrido = -2; result->time_sum_ms = 0.0;
                result->total_time_search_ms = 0.0; result->transfer_time_ms = 0.0; result->total_to_find_ms = 0.0;

               
                printf("shot %d: found=%d, found_weight=%d, degenerate=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
                shot + 1, result->found, result->found_weight, result->degenerate, result->iteration,
                result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);
    
                continue;
            }

            cudaMemcpy(d_s, detector, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_active_rows, detector_pos, weight_syndrome * sizeof(int), cudaMemcpyHostToDevice);



            if (config->use_gpu_sum) {
                cudaEventRecord(t_sum_1);
                int blocks_sum = (m + config->threads_per_block - 1) / config->threads_per_block;
                sum_columns_H_gpu<<<blocks_sum, config->threads_per_block>>>(d_H, d_active_rows, weight_syndrome, d_sum_cols, m);
                cudaEventRecord(t_sum_2);
                cudaEventSynchronize(t_sum_2);
                cudaEventElapsedTime((float*)&time_sum_ms_f, t_sum_1, t_sum_2);
                time_sum_ms = (double)time_sum_ms_f;

                cudaMemcpy(sum_cols, d_sum_cols, m * sizeof(int), cudaMemcpyDeviceToHost);

            } else if (!config->use_gpu_sum) {
                int *H_red = (int*)malloc(weight_syndrome * m * sizeof(int));

                auto t1 = std::chrono::steady_clock::now();
                generate_reduce_matrix(syndrome, n, m, H_red, H);
                sum_columns_H_red(H_red, weight_syndrome, m, sum_cols);
                auto t2 = std::chrono::steady_clock::now();

                time_sum_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                free(H_red);
            }

            else{
                fprintf(stderr, "Invalid value for use_gpu_sum: %d\n", config->use_gpu_sum);
                free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(delta); free(win_combo); free(piv_cols);
                free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
                free(column_weights);
                return -1;
            }

            if (!is_regular) {
                apply_offset_to_sum_cols(sum_cols, column_weights, m, max_soma);
            }

            auto t_total_1 = std::chrono::steady_clock::now();
            while (iteration < max_soma && iteration < 3) {
                int threshold = max_soma - iteration;
                iteration++;

                h_found_idx = -1; found = 0;

                extract_positions(sum_cols, m, h_active_pos, threshold, &tam);

                if (tam > config->max_size_active_positions) {
                    tam = tam_old;
                } else {
                    tam_old = tam;
                }

                if (tam <= 0) {
                    continue;
                }

                if ((tam > config->threshold_for_hybrid && tam <= config->max_size_active_positions && config->mode_of_search == 1) || (config->mode_of_search == 0)) {
                    auto t_transfer_1 = std::chrono::steady_clock::now();
                    cudaMemcpyToSymbol(d_found_error_idx, &reset_val, sizeof(int));
                    cudaMemcpy(d_active_pos, h_active_pos, tam * sizeof(int), cudaMemcpyHostToDevice);
                    auto t_transfer_2 = std::chrono::steady_clock::now();
                    transfer_time_ms += std::chrono::duration<double, std::milli>(t_transfer_2 - t_transfer_1).count();
                    
                    number_of_threads_for_combinations(tam, &total_threads, config->max_error_weight);
                    total_tested_combinations += total_threads;

                    blocks_search = (total_threads + config->threads_per_block - 1) / config->threads_per_block;

                    cudaEventRecord(t_search_gpu_1);
                    search_errors_massively<<<blocks_search, config->threads_per_block>>>(d_H, d_s, d_active_pos, tam, n, m, config->max_error_weight);
                    cudaEventRecord(t_search_gpu_2);
                    cudaEventSynchronize(t_search_gpu_2);
                    cudaEventElapsedTime(&time_search_ms_f, t_search_gpu_1, t_search_gpu_2);
                    time_search_ms = (double)time_search_ms_f;

                    auto t_transfer_3 = std::chrono::steady_clock::now();
                    cudaMemcpyFromSymbol(&h_found_idx, d_found_error_idx, sizeof(int));
                    auto t_transfer_4 = std::chrono::steady_clock::now();
                    transfer_time_ms += std::chrono::duration<double, std::milli>(t_transfer_4 - t_transfer_3).count();
                    if(hibrido==-1){
                        hibrido = 1;
                    }
                    else{
                        hibrido = 0;
                    }
                }

                else if ((tam <= config->threshold_for_hybrid && config->mode_of_search == 1) || (config->mode_of_search == -1)) {
                    hibrido = -1;
                    auto t_search_cpu_1 = std::chrono::steady_clock::now();
                    found = search_cpu_massively(H, syndrome, h_active_pos, tam, n, m, e_est, &number_of_iterations, config->max_error_weight);
                    auto t_search_cpu_2 = std::chrono::steady_clock::now();
                    time_search_ms = std::chrono::duration<double, std::milli>(t_search_cpu_2 - t_search_cpu_1).count();
                    total_tested_combinations += number_of_iterations;
                }

                total_time_search_ms += time_search_ms;

                if (h_found_idx != -1 || found) {
                    auto t_total_2 = std::chrono::steady_clock::now();
                    total_to_find_ms = std::chrono::duration<double, std::milli>(t_total_2 - t_total_1).count();
                    break;
                }
            }


            if (h_found_idx != -1) {
                found = 1;

                cudaMemcpyFromSymbol(win_combo, d_winning_combo, GRAND_MAX_W * sizeof(int));

                for (int w = 0; w < GRAND_MAX_W; w++) {
                    if (win_combo[w] != -1) {
                        found_weight++;
                        e_est[win_combo[w]] = 1;
                    }
                }
            }

            if (found) {
                extract_positions(e_est, m, e_est_pos, 1, &found_weight); 

                for (int i = 0; i < found_weight; i++) {
                    int event_pos = e_est_pos[i];

                    for (int j = 0; j < n_logicos; j++) {
                        logical_est[j] ^= L_dem[event_pos * n_logicos + j];
                    }
                }

                degenerate = 1; 
                for (int j = 0; j < n_logicos; j++) {
                    logicals_in_error[j] = logical_est[j] ^ logical[j];

                    if (logicals_in_error[j] == 1) {
                        degenerate = 0;
                    }
                }

                
            }

        result->found = found; result->found_weight = found_weight; result->degenerate = degenerate; result->iteration = iteration;
        result->tam = tam; result->total_tested_combinations = total_tested_combinations; result->hibrido = hibrido; result->time_sum_ms = time_sum_ms;
        result->total_time_search_ms = total_time_search_ms; result->transfer_time_ms = transfer_time_ms; result->total_to_find_ms = total_to_find_ms;
    

        printf("shot %d: found=%d, found_weight=%d, degenerate=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
        shot + 1, result->found, result->found_weight, result->degenerate, result->iteration,
        result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);

    }


        cudaEventDestroy(t_sum_1);
        cudaEventDestroy(t_sum_2);
        cudaEventDestroy(t_search_gpu_1);
        cudaEventDestroy(t_search_gpu_2);

        cudaFree(d_s); cudaFree(d_sum_cols); cudaFree(d_active_pos);cudaFree(d_H); cudaFree(d_active_rows); 

        free(H); free(syndrome); free(H_simplificada); free(sum_cols); free(h_active_pos); free(delta); 
        free(win_combo); free(piv_cols); free(e_est_pos); free(s_est); free(s_est_pos); free(e_pos); free(e_est); free(column_weights); free(logical_est); free(logicals_in_error);

        return 0;

    }

extern "C" int grand_decode_prob_cln(int *H_dem, const int *L_dem, int n, int m, int n_logicos, int *detectors, int *logicals, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results, const char *probability_file) {

        setup_nCr_table_internal();

        int *H = (int *) malloc(n * m * sizeof(int));
        memcpy(H, H_dem, n * m * sizeof(int));  
        int *H_simplificada = (int *) malloc(n * m * sizeof(int));

        int *e_pos = (int *) malloc(m * sizeof(int));
        int *e_est_pos = (int *) malloc(m * sizeof(int));
        int *e_est = (int *) malloc(m * sizeof(int));

        int *syndrome = (int *) malloc(n * sizeof(int));
        int *detector_pos = (int *) malloc(n * sizeof(int));
        int *s_est = (int *) malloc(n * sizeof(int));
        int *s_est_pos = (int *) malloc(n * sizeof(int));

        int *sum_cols = (int *) malloc(m * sizeof(int));
        int *h_active_pos = (int *) malloc(m * sizeof(int));

        int *delta = (int *) malloc(m * sizeof(int));
        int *piv_cols = (int *) malloc(n * sizeof(int));
        int *win_combo = (int *) malloc(GRAND_MAX_W * sizeof(int));

        int *logical_est = (int *) calloc(n_logicos, sizeof(int));
        int *logicals_in_error = (int *) calloc(n_logicos, sizeof(int));

        int *column_weights = (int *) malloc(m * sizeof(int));

        int *d_s, *d_sum_cols, *d_active_pos, *d_H, *d_active_rows, *d_phase_weights;

        cudaEvent_t t_sum_1, t_sum_2, t_search_1, t_search_2;

        PhaseOrderMask masks[GRAND_MAX_MASKS];

        int num_masks = 0, reset_val = -1, h_found_idx = -1, found = 0, found_weight = 0, degenerate = 0, max_soma = 0, is_regular = 0;
        int iteration = 0, tam = 0, tam_old = 0, weight_syndrome = 0, hibrido = -2, number_of_iterations = 0, shot = 0, num_phases = 0;
        unsigned long long total_tested_combinations = 0;

        double time_sum_ms = 0.0, total_time_search_ms = 0.0, transfer_time_ms = 0.0, total_to_find_ms = 0.0;
        float time_sum_ms_f = 0.0f;

        if (config->max_error_weight <= 0 || config->max_error_weight > 7) {
            free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(delta); free(win_combo); free(piv_cols);
            free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
            free(column_weights); free(logical_est); free(logicals_in_error);
            fprintf(stderr, "Invalid max_error_weight: %d. Must be between 1 and 7.\n", config->max_error_weight);
            return -1;
        }

        compute_column_stats(H_dem, n, m, column_weights, &max_soma, &is_regular);
        num_masks = read_phase_orders_from_file(probability_file, masks, GRAND_MAX_MASKS);

        if (num_masks == 0) {
            free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(delta); free(win_combo); free(piv_cols);
            free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
            free(column_weights); free(logical_est); free(logicals_in_error);
            fprintf(stderr, "No phase order masks read from: %s\n", probability_file);
            return -1;
        }

        if (config->max_error_weight == 6) {
            num_phases = 2;
        } else if (config->max_error_weight == 7) {
            num_phases = 3;
        } else if (config->max_error_weight < 6 && config->max_error_weight > 0) {
            num_phases = 1;
        } else {
            free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(delta); free(win_combo); free(piv_cols);
            free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
            free(column_weights); free(logical_est); free(logicals_in_error);
            fprintf(stderr, "Invalid max_error_weight: %d. Must be 6 or 7.\n", config->max_error_weight);
            return -1;
        }

        cudaMalloc(&d_s, n * sizeof(int));
        cudaMalloc(&d_sum_cols, m * sizeof(int));
        cudaMalloc(&d_active_pos, m * sizeof(int));
        cudaMalloc(&d_H, n * m * sizeof(int));
        cudaMalloc(&d_active_rows, n * sizeof(int));
        cudaMalloc(&d_phase_weights, GRAND_MAX_W * sizeof(int));

        cudaEventCreate(&t_sum_1);
        cudaEventCreate(&t_sum_2);
        cudaEventCreate(&t_search_1);
        cudaEventCreate(&t_search_2);

        cudaMemcpy(d_H, H, n * m * sizeof(int), cudaMemcpyHostToDevice);


        for (shot = 0; shot < num_shots; shot++) {

            GrandDecoderResult *result = &results[shot];

            memset(e_est, 0, m * sizeof(int)); memset(syndrome, 0, n * sizeof(int)); memset(result, 0, sizeof(*result)); memset(e_pos, 0, m * sizeof(int)); memset(e_est_pos, 0, m * sizeof(int)); memset(s_est, 0, n * sizeof(int));
            memset(logical_est, 0, n_logicos * sizeof(int)); memset(logicals_in_error, 0, n_logicos * sizeof(int));
            found = 0, found_weight = 0, degenerate = 0, iteration = 0, tam = 0, tam_old = 0, weight_syndrome = 0, hibrido = -2, number_of_iterations = 0;
            total_tested_combinations = 0;
            time_sum_ms = 0.0, total_time_search_ms = 0.0, transfer_time_ms = 0.0, total_to_find_ms = 0.0;
            time_sum_ms_f = 0.0f;

            int *detector = NULL;
            int *logical = NULL;

            detector = &detectors[shot * n]; 
            logical = &logicals[shot * n_logicos];

            memcpy(syndrome, detector, n * sizeof(int)); 
            extract_positions(detector, n, detector_pos, 1, &weight_syndrome);

            if (weight_syndrome == 0) {

                result->found = 0; result->found_weight = 0; result->degenerate = 0; result->iteration = 0;
                result->tam = 0; result->total_tested_combinations = 0; result->hibrido = -2; result->time_sum_ms = 0.0;
                result->total_time_search_ms = 0.0; result->transfer_time_ms = 0.0; result->total_to_find_ms = 0.0;

                printf("shot %d: found=%d, found_weight=%d, degenerate=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
                shot + 1, result->found, result->found_weight, result->degenerate, result->iteration,
                result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);

                continue;
            }

            cudaMemcpy(d_s, detector, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_active_rows, detector_pos, weight_syndrome * sizeof(int), cudaMemcpyHostToDevice);

            if (config->use_gpu_sum) {
                cudaEventRecord(t_sum_1);
                int blocks_sum = (m + config->threads_per_block - 1) / config->threads_per_block;
                sum_columns_H_gpu<<<blocks_sum, config->threads_per_block>>>(d_H, d_active_rows, weight_syndrome, d_sum_cols, m);
                cudaEventRecord(t_sum_2);
                cudaEventSynchronize(t_sum_2);
                cudaEventElapsedTime((float*)&time_sum_ms_f, t_sum_1, t_sum_2);
                time_sum_ms = (double)time_sum_ms_f;

                cudaMemcpy(sum_cols, d_sum_cols, m * sizeof(int), cudaMemcpyDeviceToHost);

            } else if (!config->use_gpu_sum) {
                int *H_red = (int*)malloc(weight_syndrome * m * sizeof(int));

                auto t1 = std::chrono::steady_clock::now();
                generate_reduce_matrix(syndrome, n, m, H_red, H);
                sum_columns_H_red(H_red, weight_syndrome, m, sum_cols);
                auto t2 = std::chrono::steady_clock::now();

                time_sum_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                free(H_red);
            } else {
                fprintf(stderr, "Invalid value for use_gpu_sum: %d\n", config->use_gpu_sum);
                free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(delta); free(win_combo); free(piv_cols);
                free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
                free(column_weights); free(logical_est); free(logicals_in_error);
                cudaFree(d_s); cudaFree(d_sum_cols); cudaFree(d_active_pos); cudaFree(d_H); cudaFree(d_active_rows); cudaFree(d_phase_weights);
                return -1;
            }

            if (!is_regular) {
                apply_offset_to_sum_cols(sum_cols, column_weights, m, max_soma);
            }

            auto t_total_1 = std::chrono::steady_clock::now();
            while (iteration < max_soma && iteration < 3) {
                int threshold = max_soma - iteration;
                iteration++;

                h_found_idx = -1; found = 0;

                extract_positions(sum_cols, m, h_active_pos, threshold, &tam);

                if (tam > config->max_size_active_positions) {
                    tam = tam_old;
                } else {
                    tam_old = tam;
                }

                if (tam <= 0) {
                    continue;
                }

                PhaseOrderMask *pm = get_mask_for_tam(tam, masks, num_masks);

                if ((tam > config->threshold_for_hybrid && tam <= config->max_size_active_positions && config->mode_of_search == 1) || (config->mode_of_search == 0)) {
                    if (hibrido == -1) {
                        hibrido = 1;
                    } else {
                        hibrido = 0;
                    }

                    auto ttr1 = std::chrono::steady_clock::now();
                    cudaMemcpyToSymbol(d_found_error_idx, &reset_val, sizeof(int));
                    cudaMemcpy(d_active_pos, h_active_pos, tam * sizeof(int), cudaMemcpyHostToDevice);
                    auto ttr2 = std::chrono::steady_clock::now();
                    transfer_time_ms += std::chrono::duration<double, std::milli>(ttr2 - ttr1).count();

                    for (int k = 0; k < num_phases; k++) {
                        int phase_id = pm->phase_order[k];
                        int h_phase_weights[GRAND_MAX_W];
                        int len_phase = 0;
                        unsigned long long phase_threads = 0;

                        get_weights_for_phase_id(phase_id, tam, h_phase_weights, &len_phase);
                        if (len_phase == 0) continue;

                        number_of_threads_for_combinations_prob(tam, h_phase_weights, len_phase, &phase_threads);
                        if (phase_threads == 0) continue;

                        total_tested_combinations += phase_threads;

                        auto ttr3 = std::chrono::steady_clock::now();
                        cudaMemcpy(d_phase_weights, h_phase_weights, len_phase * sizeof(int), cudaMemcpyHostToDevice);
                        auto ttr4 = std::chrono::steady_clock::now();
                        transfer_time_ms += std::chrono::duration<double, std::milli>(ttr4 - ttr3).count();

                        unsigned int blocks_search = (unsigned int)((phase_threads + config->threads_per_block - 1) / config->threads_per_block);

                        cudaEventRecord(t_search_1);
                        search_errors_massively_prob<<<blocks_search, config->threads_per_block>>>( d_H, d_s, d_active_pos, tam, n, m, d_phase_weights, len_phase );
                        cudaEventRecord(t_search_2);
                        cudaEventSynchronize(t_search_2);

                        float phase_search_ms = 0.0f;
                        cudaEventElapsedTime(&phase_search_ms, t_search_1, t_search_2);
                        total_time_search_ms += phase_search_ms;

                        auto ttr5 = std::chrono::steady_clock::now();
                        cudaMemcpyFromSymbol(&h_found_idx, d_found_error_idx, sizeof(int));
                        auto ttr6 = std::chrono::steady_clock::now();
                        transfer_time_ms += std::chrono::duration<double, std::milli>(ttr6 - ttr5).count();

                        if (h_found_idx != -1) {
                            break;
                        }
                    }
                } else if ((tam <= config->threshold_for_hybrid && config->mode_of_search == 1) || (config->mode_of_search == -1)) {
                    hibrido = -1;

                    auto tc1 = std::chrono::steady_clock::now();

                    for (int k = 0; k < num_phases; k++) {
                        int phase_id = pm->phase_order[k];
                        int h_phase_weights[GRAND_MAX_W];
                        int len_phase = 0;
                        int iterations_phase = 0;

                        get_weights_for_phase_id(phase_id, tam, h_phase_weights, &len_phase);
                        if (len_phase == 0) continue;

                        found = search_cpu_phase(H, syndrome, h_active_pos, tam, n, m, e_est, &iterations_phase, h_phase_weights, len_phase);

                        total_tested_combinations += (unsigned long long)iterations_phase;

                        if (found) {
                            break;
                        }
                    }

                    auto tc2 = std::chrono::steady_clock::now();
                    total_time_search_ms += std::chrono::duration<double, std::milli>(tc2 - tc1).count();
                }

                if (h_found_idx != -1 || found) {
                    auto t_total_2 = std::chrono::steady_clock::now();
                    total_to_find_ms = std::chrono::duration<double, std::milli>(t_total_2 - t_total_1).count();
                    break;
                }
            }

            if (h_found_idx != -1) {
                found = 1;

                cudaMemcpyFromSymbol(win_combo, d_winning_combo, GRAND_MAX_W * sizeof(int));

                for (int w = 0; w < GRAND_MAX_W; w++) {
                    if (win_combo[w] != -1) {
                        found_weight++;
                        e_est[win_combo[w]] = 1;
                    }
                }
            }

            if (found) {
                extract_positions(e_est, m, e_est_pos, 1, &found_weight);

                for (int i = 0; i < found_weight; i++) {
                    int event_pos = e_est_pos[i];

                    for (int j = 0; j < n_logicos; j++) {
                        logical_est[j] ^= L_dem[event_pos * n_logicos + j];
                    }
                }

                degenerate = 1;
                for (int j = 0; j < n_logicos; j++) {
                    logicals_in_error[j] = logical_est[j] ^ logical[j];

                    if (logicals_in_error[j] == 1) {
                        degenerate = 0;
                    }
                }
            }

            result->found = found; result->found_weight = found_weight; result->degenerate = degenerate; result->iteration = iteration;
            result->tam = tam; result->total_tested_combinations = total_tested_combinations; result->hibrido = hibrido; result->time_sum_ms = time_sum_ms;
            result->total_time_search_ms = total_time_search_ms; result->transfer_time_ms = transfer_time_ms; result->total_to_find_ms = total_to_find_ms;

            printf("shot %d: found=%d, found_weight=%d, degenerate=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
            shot + 1, result->found, result->found_weight, result->degenerate, result->iteration,
            result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);
        }

        cudaEventDestroy(t_sum_1);
        cudaEventDestroy(t_sum_2);
        cudaEventDestroy(t_search_1);
        cudaEventDestroy(t_search_2);

        cudaFree(d_s); cudaFree(d_sum_cols); cudaFree(d_active_pos); cudaFree(d_H); cudaFree(d_active_rows); cudaFree(d_phase_weights);

        free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(delta);
        free(win_combo); free(piv_cols); free(e_est_pos); free(s_est); free(s_est_pos); free(e_pos); free(e_est); free(column_weights); free(logical_est); free(logicals_in_error);

        return 0;

    }

extern "C" int grand_decode_ML_cln(int *H_dem, const int *L_dem, int n, int m, int n_logicos, int *detectors, int *logicals, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results, const char *llr_file) {

        setup_nCr_table_internal();

        int *H = (int *) malloc(n * m * sizeof(int));
        memcpy(H, H_dem, n * m * sizeof(int));
        int *H_simplificada = (int *) malloc(n * m * sizeof(int));

        int *e_pos = (int *) malloc(m * sizeof(int));
        int *e_est_pos = (int *) malloc(m * sizeof(int));
        int *e_est = (int *) malloc(m * sizeof(int));

        int *syndrome = (int *) malloc(n * sizeof(int));
        int *detector_pos = (int *) malloc(n * sizeof(int));
        int *s_est = (int *) malloc(n * sizeof(int));
        int *s_est_pos = (int *) malloc(n * sizeof(int));

        int *sum_cols = (int *) malloc(m * sizeof(int));
        int *h_active_pos = (int *) malloc(m * sizeof(int));

        int *delta = (int *) malloc(m * sizeof(int));
        int *piv_cols = (int *) malloc(n * sizeof(int));
        int *win_combo = (int *) malloc(GRAND_MAX_W * sizeof(int));

        int *logical_est = (int *) calloc(n_logicos, sizeof(int));
        int *logicals_in_error = (int *) calloc(n_logicos, sizeof(int));

        int *column_weights = (int *) malloc(m * sizeof(int));

        float *llr = (float *) malloc(m * sizeof(float));

        int *d_s, *d_sum_cols, *d_active_pos, *d_H, *d_active_rows;
        float *d_llr;

        cudaEvent_t t_sum_1, t_sum_2, t_search_gpu_1, t_search_gpu_2;

        int reset_val = -1, h_found_idx = -1, found = 0, found_weight = 0, degenerate = 0, max_soma = 0, is_regular = 0;
        int iteration = 0, tam = 0, tam_old = 0, weight_syndrome = 0, hibrido = -2, number_of_iterations = 0, shot = 0;
        unsigned long long total_tested_combinations = 0, total_threads = 0;
        unsigned int blocks_search;

        double time_sum_ms = 0.0, total_time_search_ms = 0.0, transfer_time_ms = 0.0, total_to_find_ms = 0.0, time_search_ms = 0.0;
        float time_sum_ms_f = 0.0f, time_search_ms_f = 0.0f;

        if (config->max_error_weight <= 0 || config->max_error_weight > 7) {
            free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(delta); free(win_combo); free(piv_cols);
            free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
            free(column_weights); free(logical_est); free(logicals_in_error); free(llr);
            fprintf(stderr, "Invalid max_error_weight: %d. Must be between 1 and 7.\n", config->max_error_weight);
            return -1;
        }

        if (!read_llr_from_file_ml(llr_file, llr, m)) {
            free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(delta); free(win_combo); free(piv_cols);
            free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
            free(column_weights); free(logical_est); free(logicals_in_error); free(llr);
            return -1;
        }

        compute_column_stats(H_dem, n, m, column_weights, &max_soma, &is_regular);

        cudaMalloc(&d_s, n * sizeof(int));
        cudaMalloc(&d_sum_cols, m * sizeof(int));
        cudaMalloc(&d_active_pos, m * sizeof(int));
        cudaMalloc(&d_H, n * m * sizeof(int));
        cudaMalloc(&d_active_rows, n * sizeof(int));
        cudaMalloc(&d_llr, m * sizeof(float));

        cudaEventCreate(&t_sum_1);
        cudaEventCreate(&t_sum_2);
        cudaEventCreate(&t_search_gpu_1);
        cudaEventCreate(&t_search_gpu_2);

        cudaMemcpy(d_H, H, n * m * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_llr, llr, m * sizeof(float), cudaMemcpyHostToDevice);


        for (shot = 0; shot < num_shots; shot++) {

            GrandDecoderResult *result = &results[shot];

            memset(e_est, 0, m * sizeof(int)); memset(syndrome, 0, n * sizeof(int)); memset(result, 0, sizeof(*result)); memset(e_pos, 0, m * sizeof(int)); memset(e_est_pos, 0, m * sizeof(int)); memset(s_est, 0, n * sizeof(int));
            memset(logical_est, 0, n_logicos * sizeof(int)); memset(logicals_in_error, 0, n_logicos * sizeof(int));
            found = 0, found_weight = 0, degenerate = 0, iteration = 0, tam = 0, tam_old = 0, weight_syndrome = 0, hibrido = -2, number_of_iterations = 0;
            total_tested_combinations = 0, total_threads = 0, blocks_search = 0;
            time_sum_ms = 0.0, total_time_search_ms = 0.0, transfer_time_ms = 0.0, total_to_find_ms = 0.0, time_search_ms = 0.0;
            time_sum_ms_f = 0.0f, time_search_ms_f = 0.0f;

            int *detector = NULL;
            int *logical = NULL;

            detector = &detectors[shot * n]; 
            logical = &logicals[shot * n_logicos];

            memcpy(syndrome, detector, n * sizeof(int)); 
            extract_positions(detector, n, detector_pos, 1, &weight_syndrome);

            if (weight_syndrome == 0) {

                result->found = 0; result->found_weight = 0; result->degenerate = 0; result->iteration = 0;
                result->tam = 0; result->total_tested_combinations = 0; result->hibrido = -2; result->time_sum_ms = 0.0;
                result->total_time_search_ms = 0.0; result->transfer_time_ms = 0.0; result->total_to_find_ms = 0.0;

                printf("shot %d: found=%d, found_weight=%d, degenerate=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
                shot + 1, result->found, result->found_weight, result->degenerate, result->iteration,
                result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);

                continue;
            }

            cudaMemcpy(d_s, detector, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_active_rows, detector_pos, weight_syndrome * sizeof(int), cudaMemcpyHostToDevice);

            if (config->use_gpu_sum) {
                cudaEventRecord(t_sum_1);
                int blocks_sum = (m + config->threads_per_block - 1) / config->threads_per_block;
                sum_columns_H_gpu<<<blocks_sum, config->threads_per_block>>>(d_H, d_active_rows, weight_syndrome, d_sum_cols, m);
                cudaEventRecord(t_sum_2);
                cudaEventSynchronize(t_sum_2);
                cudaEventElapsedTime((float*)&time_sum_ms_f, t_sum_1, t_sum_2);
                time_sum_ms = (double)time_sum_ms_f;

                cudaMemcpy(sum_cols, d_sum_cols, m * sizeof(int), cudaMemcpyDeviceToHost);

            } else if (!config->use_gpu_sum) {
                int *H_red = (int*)malloc(weight_syndrome * m * sizeof(int));

                auto t1 = std::chrono::steady_clock::now();
                generate_reduce_matrix(syndrome, n, m, H_red, H);
                sum_columns_H_red(H_red, weight_syndrome, m, sum_cols);
                auto t2 = std::chrono::steady_clock::now();

                time_sum_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                free(H_red);
            } else {
                fprintf(stderr, "Invalid value for use_gpu_sum: %d\n", config->use_gpu_sum);
                free(H); free(H_simplificada); free(sum_cols); free(h_active_pos); free(delta); free(win_combo); free(piv_cols);
                free(e_pos); free(e_est_pos); free(e_est); free(syndrome); free(s_est); free(s_est_pos);
                free(column_weights); free(logical_est); free(logicals_in_error); free(llr);
                cudaFree(d_s); cudaFree(d_sum_cols); cudaFree(d_active_pos); cudaFree(d_H); cudaFree(d_active_rows); cudaFree(d_llr);
                return -1;
            }

            if (!is_regular) {
                apply_offset_to_sum_cols(sum_cols, column_weights, m, max_soma);
            }

            auto t_total_1 = std::chrono::steady_clock::now();
            while (iteration < max_soma && iteration < 3) {
                int threshold = max_soma - iteration;
                iteration++;

                h_found_idx = -1; found = 0;

                extract_positions(sum_cols, m, h_active_pos, threshold, &tam);

                if (tam > config->max_size_active_positions) {
                    tam = tam_old;
                } else {
                    tam_old = tam;
                }

                if (tam <= 0) {
                    continue;
                }

                if ((tam > config->threshold_for_hybrid && tam <= config->max_size_active_positions && config->mode_of_search == 1) || (config->mode_of_search == 0)) {
                    auto t_transfer_1 = std::chrono::steady_clock::now();
                    unsigned long long reset_key = 0xFFFFFFFFFFFFFFFFULL;
                    cudaMemcpyToSymbol(d_best_key_ml, &reset_key, sizeof(reset_key));
                    cudaMemcpy(d_active_pos, h_active_pos, tam * sizeof(int), cudaMemcpyHostToDevice);
                    auto t_transfer_2 = std::chrono::steady_clock::now();
                    transfer_time_ms += std::chrono::duration<double, std::milli>(t_transfer_2 - t_transfer_1).count();

                    number_of_threads_for_combinations(tam, &total_threads, config->max_error_weight);
                    total_tested_combinations += total_threads;

                    blocks_search = (total_threads + config->threads_per_block - 1) / config->threads_per_block;

                    cudaEventRecord(t_search_gpu_1);
                    search_errors_massively_ml<<<blocks_search, config->threads_per_block>>>(d_H, d_s, d_active_pos, tam, n, m, config->max_error_weight, d_llr);
                    cudaEventRecord(t_search_gpu_2);
                    cudaEventSynchronize(t_search_gpu_2);
                    cudaEventElapsedTime(&time_search_ms_f, t_search_gpu_1, t_search_gpu_2);
                    time_search_ms = (double)time_search_ms_f;

                    auto t_transfer_3 = std::chrono::steady_clock::now();
                    unsigned long long h_best_key;
                    cudaMemcpyFromSymbol(&h_best_key, d_best_key_ml, sizeof(h_best_key));
                    auto t_transfer_4 = std::chrono::steady_clock::now();
                    transfer_time_ms += std::chrono::duration<double, std::milli>(t_transfer_4 - t_transfer_3).count();

                    if (h_best_key != 0xFFFFFFFFFFFFFFFFULL) {
                        unsigned long long winning_tid = h_best_key & 0xFFFFFFFFULL;
                        int winning_weight;
                        int winning_combo[GRAND_MAX_W];
                        decode_winning_combo_ml(winning_tid, tam, config->max_error_weight, h_active_pos, &winning_weight, winning_combo);

                        h_found_idx = 1;
                        for (int w = 0; w < GRAND_MAX_W; w++) win_combo[w] = winning_combo[w];
                    }

                    if(hibrido==-1){
                        hibrido = 1;
                    }
                    else{
                        hibrido = 0;
                    }
                }

                else if ((tam <= config->threshold_for_hybrid && config->mode_of_search == 1) || (config->mode_of_search == -1)) {
                    hibrido = -1;
                    auto t_search_cpu_1 = std::chrono::steady_clock::now();
                    found = search_cpu_ml(H, syndrome, h_active_pos, tam, n, m, e_est, &number_of_iterations, config->max_error_weight, llr);
                    auto t_search_cpu_2 = std::chrono::steady_clock::now();
                    time_search_ms = std::chrono::duration<double, std::milli>(t_search_cpu_2 - t_search_cpu_1).count();
                    total_tested_combinations += number_of_iterations;
                }

                total_time_search_ms += time_search_ms;

                if (h_found_idx != -1 || found) {
                    auto t_total_2 = std::chrono::steady_clock::now();
                    total_to_find_ms = std::chrono::duration<double, std::milli>(t_total_2 - t_total_1).count();
                    break;
                }
            }

            if (h_found_idx != -1) {
                found = 1;

                for (int w = 0; w < GRAND_MAX_W; w++) {
                    if (win_combo[w] != -1) {
                        found_weight++;
                        e_est[win_combo[w]] = 1;
                    }
                }
            }

            if (found) {
                extract_positions(e_est, m, e_est_pos, 1, &found_weight);

                for (int i = 0; i < found_weight; i++) {
                    int event_pos = e_est_pos[i];

                    for (int j = 0; j < n_logicos; j++) {
                        logical_est[j] ^= L_dem[event_pos * n_logicos + j];
                    }
                }

                degenerate = 1;
                for (int j = 0; j < n_logicos; j++) {
                    logicals_in_error[j] = logical_est[j] ^ logical[j];

                    if (logicals_in_error[j] == 1) {
                        degenerate = 0;
                    }
                }
            }

            result->found = found; result->found_weight = found_weight; result->degenerate = degenerate; result->iteration = iteration;
            result->tam = tam; result->total_tested_combinations = total_tested_combinations; result->hibrido = hibrido; result->time_sum_ms = time_sum_ms;
            result->total_time_search_ms = total_time_search_ms; result->transfer_time_ms = transfer_time_ms; result->total_to_find_ms = total_to_find_ms;

            printf("shot %d: found=%d, found_weight=%d, degenerate=%d, iteration=%d, tam=%d, total_tested_combinations=%llu, hybrid=%d, time_sum_ms=%.5f, total_time_search_ms=%.5f, transfer_time_ms=%.5f, total_to_find_ms=%.5f\n",
            shot + 1, result->found, result->found_weight, result->degenerate, result->iteration,
            result->tam, result->total_tested_combinations, result->hibrido, result->time_sum_ms, result->total_time_search_ms, result->transfer_time_ms, result->total_to_find_ms);
        }

        cudaEventDestroy(t_sum_1);
        cudaEventDestroy(t_sum_2);
        cudaEventDestroy(t_search_gpu_1);
        cudaEventDestroy(t_search_gpu_2);

        cudaFree(d_s); cudaFree(d_sum_cols); cudaFree(d_active_pos); cudaFree(d_H); cudaFree(d_active_rows); cudaFree(d_llr);

        free(H); free(syndrome); free(H_simplificada); free(sum_cols); free(h_active_pos); free(delta);
        free(win_combo); free(piv_cols); free(e_est_pos); free(s_est); free(s_est_pos); free(e_pos); free(e_est); free(column_weights); free(logical_est); free(logicals_in_error); free(llr);

        return 0;

    }
