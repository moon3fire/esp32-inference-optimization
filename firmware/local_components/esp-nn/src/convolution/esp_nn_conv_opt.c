// Copyright 2020-2021 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <esp_nn_defs.h>
#include <esp_nn_ansi_headers.h>

#include <common_functions.h>

int esp_nn_get_conv_scratch_size_opt(const data_dims_t *input_dims,
                                     const data_dims_t *filter_dims,
                                     const data_dims_t *output_dims,
                                     const conv_params_t *conv_params)
{
    return 0;
}

void esp_nn_set_conv_scratch_buf_opt(const void *buf)
{

}

__attribute__ ((noinline))
static void esp_nn_conv_s8_1x1(const data_dims_t *input_dims,
                               const int8_t *input_data,
                               const int8_t *filter_data,
                               const int32_t *bias,
                               const data_dims_t *output_dims,
                               int8_t *out_data,
                               const conv_params_t *conv_params,
                               const quant_data_t *quant_data)
{
    const uint16_t input_wd = input_dims->width;
    const uint16_t in_channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_channels = output_dims->channels;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    for (int32_t in_row = 0; in_row < out_ht * stride_ht; in_row += stride_ht) {
        for (int32_t in_col = 0; in_col < out_wd * stride_wd; in_col += stride_wd) {
            const int32_t *out_mult = quant_data->mult;
            const int32_t *out_shift = quant_data->shift;
            const int8_t *filter_ptr = filter_data;
            const int8_t *input_base_ptr = input_data + (in_row * input_wd + in_col) * in_channels;
            int32_t out_ch_idx = 0;
            for (; out_ch_idx < out_channels; out_ch_idx++) {
                int32_t conv_out = 0;

                const int8_t *input_ptr = input_base_ptr;

                int32_t in_ch_idx = 0;
                for (; in_ch_idx < in_channels - 3; in_ch_idx += 4) {
                    conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                    conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                    conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                    conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                }
                for (; in_ch_idx < in_channels; in_ch_idx ++) {
                    conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                }
                if (bias) {
                    conv_out += bias[out_ch_idx];
                }
                conv_out = esp_nn_requantize(conv_out, *out_mult++, *out_shift++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_data++ = (int8_t) conv_out;
            }
        }
    }
}

/**
 * Assumption 1: i/p channels == o/p channels
 * Assumption 2: Pointers are valid
 * Assumption 3: dialation width = 1
 */
__attribute__((noipa))
static void edgeinfer_conv_original(const data_dims_t *input_dims,
                        const int8_t *input_data,
                        const data_dims_t *filter_dims,
                        const int8_t *filter_data,
                        const int32_t *bias,
                        const data_dims_t *output_dims,
                        int8_t *out_data,
                        const conv_params_t *conv_params,
                        const quant_data_t *quant_data)
{
    const uint16_t filter_wd = filter_dims->width;
    const uint16_t filter_ht = filter_dims->height;

    /* Grouped conv (filter_ch < input_ch) must be caught BEFORE the 1x1
     * fast path: a grouped 1x1 would otherwise run at full input depth and
     * produce wrong results. The S3/PIE dispatches already catch this. */
    if (input_dims->channels != filter_dims->channels) {
        esp_nn_conv_s8_ansi(input_dims, input_data, filter_dims, filter_data,
                            bias, output_dims, out_data, conv_params, quant_data);
        return;
    }

    if (filter_wd == 1 && filter_ht == 1) {
        esp_nn_conv_s8_1x1(input_dims, input_data, filter_data, bias,
                           output_dims, out_data, conv_params, quant_data);
        return;
    }

    const uint16_t input_wd = input_dims->width;
    const uint16_t input_ht = input_dims->height;
    const uint16_t in_channels = input_dims->channels;
    const int32_t input_offset = conv_params->in_offset;
    const int32_t out_offset = conv_params->out_offset;
    const uint16_t pad_wd = conv_params->padding.width;
    const uint16_t pad_ht = conv_params->padding.height;
    const uint16_t stride_wd = conv_params->stride.width;
    const uint16_t stride_ht = conv_params->stride.height;
    const uint16_t out_wd = output_dims->width;
    const uint16_t out_ht = output_dims->height;
    const uint16_t out_channels = output_dims->channels;
    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    /* Grouped conv (filter_ch < input_ch): fall back to ansi which handles it */
    if (in_channels != filter_dims->channels) {
        esp_nn_conv_s8_ansi(input_dims, input_data, filter_dims, filter_data,
                            bias, output_dims, out_data, conv_params, quant_data);
        return;
    }

    int32_t out_ch_idx, out_y, out_x, filter_y_idx, filter_x_idx;

    for (out_y = 0; out_y < out_ht; out_y++) {
        for (out_x = 0; out_x < out_wd; out_x++) {
            const int32_t *out_shift = quant_data->shift;
            const int32_t *out_mult = quant_data->mult;
            for (out_ch_idx = 0; out_ch_idx < out_channels; out_ch_idx++) {
                int32_t conv_out = 0;

                const int32_t base_y = stride_ht * out_y - pad_ht;
                const int32_t base_x = stride_wd * out_x - pad_wd;

                const int32_t filter_y_start = max(0, -base_y);
                const int32_t filter_x_start = max(0, -base_x);

                const int32_t filter_y_end = min(filter_ht, input_ht - base_y);
                const int32_t filter_x_end = min(filter_wd, input_wd - base_x);

                for (filter_y_idx = filter_y_start; filter_y_idx < filter_y_end; filter_y_idx++) {
                    for (filter_x_idx = filter_x_start; filter_x_idx < filter_x_end; filter_x_idx++) {
                        const int32_t in_row = base_y + filter_y_idx;
                        const int32_t in_col = base_x + filter_x_idx;

                        const int8_t *input_ptr = input_data +
                                        (in_row * input_wd + in_col) * in_channels;
                        const int8_t *filter_ptr = filter_data +
                                        out_ch_idx * in_channels * filter_ht * filter_wd +
                                        (filter_y_idx * filter_wd + filter_x_idx) * in_channels;
                        int32_t in_ch_idx = 0;
                        for (; in_ch_idx < in_channels - 3; in_ch_idx += 4) {
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                        }
                        for (; in_ch_idx < in_channels; in_ch_idx ++) {
                            conv_out += (*input_ptr++ + input_offset) * *filter_ptr++;
                        }
                    }
                }
                if (bias) {
                    conv_out += bias[out_ch_idx];
                }
                conv_out = esp_nn_requantize(conv_out, *out_mult++, *out_shift++);
                conv_out += out_offset;
                conv_out = max(conv_out, activation_min);
                conv_out = min(conv_out, activation_max);
                *out_data++ = (int8_t) conv_out;
            }
        }
    }
}


/*
 * EdgeInfer experiment:
 * Specialized first convolution for the current MNIST model.
 * All other configurations use the original ESP-NN implementation.
 *
 * Verification mode executes both implementations. Disable it only
 * after checking the intermediate outputs on the board.
 */
#define EDGEINFER_CUSTOM_CONV1 1
#define EDGEINFER_VERIFY_CONV1 0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int edgeinfer_matches_conv1(
    const data_dims_t *input_dims,
    const data_dims_t *filter_dims,
    const data_dims_t *output_dims,
    const conv_params_t *params)
{
    return input_dims->width == 28 &&
           input_dims->height == 28 &&
           input_dims->channels == 1 &&
           filter_dims->width == 3 &&
           filter_dims->height == 3 &&
           filter_dims->channels == 1 &&
           output_dims->width == 26 &&
           output_dims->height == 26 &&
           output_dims->channels == 8 &&
           params->stride.width == 1 &&
           params->stride.height == 1 &&
           params->padding.width == 0 &&
           params->padding.height == 0 &&
           params->in_offset == 128 &&
           params->dilation.width == 0 &&
           params->dilation.height == 0;
}

static void edgeinfer_conv1(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int8_t *output,
    const conv_params_t *params,
    const quant_data_t *quant)
{
    for (int y = 0; y < 26; ++y) {
        for (int x = 0; x < 26; ++x) {
            const int8_t *patch = input + y * 28 + x;

            /*
             * Correct the input zero point once per patch.
             * Each value is in [0, 255].
             */
            const int32_t p0 = (int32_t)patch[0]  + 128;
            const int32_t p1 = (int32_t)patch[1]  + 128;
            const int32_t p2 = (int32_t)patch[2]  + 128;
            const int32_t p3 = (int32_t)patch[28] + 128;
            const int32_t p4 = (int32_t)patch[29] + 128;
            const int32_t p5 = (int32_t)patch[30] + 128;
            const int32_t p6 = (int32_t)patch[56] + 128;
            const int32_t p7 = (int32_t)patch[57] + 128;
            const int32_t p8 = (int32_t)patch[58] + 128;

            for (int channel = 0; channel < 8; ++channel) {
                const int8_t *w = weights + channel * 9;

                int32_t sum = 0;
                sum += p0 * w[0];
                sum += p1 * w[1];
                sum += p2 * w[2];
                sum += p3 * w[3];
                sum += p4 * w[4];
                sum += p5 * w[5];
                sum += p6 * w[6];
                sum += p7 * w[7];
                sum += p8 * w[8];

                if (bias != NULL) {
                    sum += bias[channel];
                }

                // Preserve the original ESP-NN quantization semantics.
                sum = esp_nn_requantize(
                    sum,
                    quant->mult[channel],
                    quant->shift[channel]);

                sum += params->out_offset;

                if (sum < params->activation.min) {
                    sum = params->activation.min;
                }
                if (sum > params->activation.max) {
                    sum = params->activation.max;
                }

                *output++ = (int8_t)sum;
            }
        }
    }
}

void esp_nn_conv_s8_opt(
    const data_dims_t *input_dims,
    const int8_t *input_data,
    const data_dims_t *filter_dims,
    const int8_t *filter_data,
    const int32_t *bias,
    const data_dims_t *output_dims,
    int8_t *out_data,
    const conv_params_t *conv_params,
    const quant_data_t *quant_data)
{
    if (!EDGEINFER_CUSTOM_CONV1 || !edgeinfer_matches_conv1(
            input_dims, filter_dims, output_dims, conv_params)) {
        edgeinfer_conv_original(
            input_dims, input_data,
            filter_dims, filter_data,
            bias, output_dims, out_data,
            conv_params, quant_data);
        return;
    }

    edgeinfer_conv1(
        input_data, filter_data, bias, out_data,
        conv_params, quant_data);

#if EDGEINFER_VERIFY_CONV1
    /*
     * Extra RAM for verification only.
     * This single-interpreter test runs sequentially.
     */
    static int8_t reference[26 * 26 * 8];
    static int announced = 0;

    edgeinfer_conv_original(
        input_dims, input_data,
        filter_dims, filter_data,
        bias, output_dims, reference,
        conv_params, quant_data);

    if (memcmp(out_data, reference, sizeof(reference)) != 0) {
        for (size_t i = 0; i < sizeof(reference); ++i) {
            if (out_data[i] != reference[i]) {
                printf(
                    "CONV1 FAIL at output %u: custom=%d original=%d\n",
                    (unsigned)i,
                    (int)out_data[i],
                    (int)reference[i]);
                break;
            }
        }
        abort();
    }

    if (!announced) {
        printf(
            "CONV1 custom path active: first 5408 outputs match exactly.\n"
            "Verification remains enabled for every subsequent call.\n");
        announced = 1;
    }
#endif
}
