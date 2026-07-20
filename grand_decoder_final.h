#ifndef GRAND_DECODER_H
#define GRAND_DECODER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>


#define GRAND_MAX_W 6


typedef struct {
    int max_error_weight;
    int threads_per_block;
    int use_gpu_sum;
    int max_size_active_positions;
    int threshold_for_hybrid;
    int mode_of_search; // -1 = cpu_search, 0 = gpu_search, 1 = hybrid_search
} GrandDecoderConfig;


typedef struct {
    int found;
    int found_weight;
    int original_weight;
    int degenerate;
    int is_equal;
    int tam;
    unsigned long long total_tested_combinations;
    int hibrido;
    int iteration;

    double time_sum_ms;
    double total_time_search_ms;
    double transfer_time_ms;
    double total_to_find_ms;
} GrandDecoderResult;



extern "C" int grand_decode_prob_cc(const int *Hx, const int *Hz, int n, int m, int *errors, int *syndromes, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results, char type_of_error, const char *probability_file);
 
extern "C" int grand_decode_cc(const int *Hx, const int *Hz, int n, int m, int *errors, int *syndromes, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results, char type_of_error);

extern "C" int grand_decode_cln(int *H_dem, const int *L_dem, int n, int m, int n_logicos, int *detectors, int *logicals, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results);

extern "C" int grand_decode_prob_cln(int *H_dem, const int *L_dem, int n, int m, int n_logicos, int *detectors, int *logicals, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results, const char *probability_file);

extern "C" int grand_decode_ML_cln(int *H_dem, const int *L_dem, int n, int m, int n_logicos, int *detectors, int *logicals, int num_shots, const GrandDecoderConfig *config, GrandDecoderResult *results, const char *llr_file);


#ifdef __cplusplus
}
#endif

#endif
