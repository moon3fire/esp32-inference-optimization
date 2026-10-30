#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../tools/esp_nn_conv_opt.custom.c"

static uint32_t state = 42;
static uint32_t random_word(void) {
    state = state * 1664525u + 1013904223u;
    return state;
}

int main(void) {
    int8_t input[28 * 28], weights[8 * 9];
    int8_t expected[26 * 26 * 8], actual[26 * 26 * 8];
    int32_t biases[8], shifts[8], multipliers[8];
    data_dims_t in = {28, 28, 1, 1};
    data_dims_t filter = {3, 3, 1, 0};
    data_dims_t out = {26, 26, 8, 1};
    conv_params_t params = {128, 0, {1, 1}, {0, 0}, {0, 0}, {-128, 127}};
    quant_data_t quant = {shifts, multipliers};

    for (int trial = 0; trial < 128; ++trial) {
        for (unsigned i = 0; i < sizeof(input); ++i)
            input[i] = (int8_t)((int)(random_word() >> 24) - 128);
        for (unsigned i = 0; i < sizeof(weights); ++i)
            weights[i] = (int8_t)((int)(random_word() >> 24) - 128);
        if (trial == 0) memset(input, -128, sizeof(input));
        if (trial == 1) memset(input, 127, sizeof(input));
        for (int c = 0; c < 8; ++c) {
            biases[c] = (int32_t)(random_word() % 40001u) - 20000;
            shifts[c] = -7 - (int32_t)(random_word() % 6u);
            multipliers[c] = 1073741824 + (int32_t)(random_word() % 1000000000u);
        }
        params.out_offset = (trial % 31) - 15;
        params.activation.min = trial % 2 ? -20 : -128;
        const int32_t *bias = trial % 3 ? biases : NULL;
        edgeinfer_conv_original(&in, input, &filter, weights, bias,
                                &out, expected, &params, &quant);
        esp_nn_conv_s8_opt(&in, input, &filter, weights, bias,
                          &out, actual, &params, &quant);
        if (memcmp(expected, actual, sizeof(actual))) {
            fprintf(stderr, "Mismatch at trial %d\n", trial);
            return 1;
        }
    }
    puts("PASS: 128 synthetic cases, 692224 output comparisons; original vs custom kernel.");
    return 0;
}
