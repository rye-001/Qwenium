#pragma once
// tensor_tracer.h — debug-only tensor inspection for graph bring-up.
//
// Responsibility: print tensor shapes/ops/values while walking a built
//   ggml_cgraph, so a new recipe's graph can be eyeballed against a reference.
//   A DIAGNOSTIC, not part of any decode path: cli/main.cpp is the only
//   consumer, and nothing in the engine calls it.
// Header-only and self-contained so it can be dropped into a scratch harness.
// No unit test by design — it exists to help write the tests.

#include "ggml.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>

// Redundant with ggml.h above, kept so this header can also be included in a
// scratch translation unit that has only the ggml forward declarations.
struct ggml_tensor;
enum ggml_op;

// Helper to print first N values based on tensor type

// Helper to print first N values based on tensor type
static void print_tensor_values(const ggml_tensor * tensor, int n) {
    if (!tensor || !tensor->data) {
        printf("ERROR: No tensor data available!\n");
        return;
    }

    printf("First %d values: ", n);

    switch (tensor->type) {
        case GGML_TYPE_F32:
            {
                float * data = (float *) tensor->data;
                for (int i = 0; i < n; i++) {
                    printf("%.6f ", data[i]);
                }
                break;
            }
        case GGML_TYPE_I32:
            {
                int32_t * data = (int32_t *) tensor->data;
                for (int i = 0; i < n; i++) {
                    printf("%d ", data[i]);
                }
                break;
            }
        case GGML_TYPE_F16:
            {
                ggml_fp16_t * data = (ggml_fp16_t *) tensor->data;
                for (int i = 0; i < n; i++) {
                    printf("%.6f ", ggml_fp16_to_fp32(data[i]));
                }
                break;
            }
        case GGML_TYPE_BF16:
            {
                ggml_bf16_t * data = (ggml_bf16_t *) tensor->data;
                for (int i = 0; i < n; i++) {
                    printf("%.6f ", ggml_bf16_to_fp32(data[i]));
                }
                break;
            }
        case GGML_TYPE_Q8_0:
            {
                // Q8_0: block of 32 int8 values + 1 fp16 scale per block
                const int block_size = 32;
                uint8_t * base_ptr   = (uint8_t *) tensor->data;

                for (int i = 0; i < n; i++) {
                    int           block_idx    = i / block_size;
                    int           in_block_idx = i % block_size;
                    size_t        block_offset = block_idx * (sizeof(ggml_fp16_t) + block_size);
                    ggml_fp16_t * scale        = (ggml_fp16_t *) (base_ptr + block_offset);
                    int8_t *      quants       = (int8_t *) (base_ptr + block_offset + sizeof(ggml_fp16_t));
                    float         scale_f      = ggml_fp16_to_fp32(*scale);
                    printf("%.6f ", scale_f * quants[in_block_idx]);
                }
                break;
            }
        default:
            printf("(unsupported type: %d)", tensor->type);
            break;
    }
    printf("\n");
}

static void debug_tensor(const ggml_tensor * tensor) {
    if (!tensor) {
        return;
    }

    printf("Shape: [%d, %d, %d, %d]\n", tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    printf("Node name: %s\n", tensor->name);
    printf("Node OP: %s\n", ggml_op_desc(tensor));
    printf("Type: %s\n", ggml_type_name(tensor->type));

    print_tensor_values(tensor, 5);

    if (tensor->src[0]) {
        printf("Src[0] name: %s, type: %s\n", tensor->src[0]->name, ggml_type_name(tensor->src[0]->type));
        print_tensor_values(tensor->src[0], 5);
    }

    if (tensor->src[1]) {
        printf("Src[1] name: %s, type: %s\n", tensor->src[1]->name, ggml_type_name(tensor->src[1]->type));
        print_tensor_values(tensor->src[1], 5);
    }

    printf("\n");
}

class TensorGraphTracer {
  private:
    std::unordered_set<const ggml_tensor *> visited;

    std::string get_op_name(ggml_op op, const ggml_tensor * tensor) {
        // Your existing get_op_name and get_unary_op_name logic combined
        switch (op) {
            case GGML_OP_NONE:
                return "DATA";
            case GGML_OP_ADD:
                return "ADD";
            case GGML_OP_MUL_MAT:
                return "MUL_MAT";
            case GGML_OP_RMS_NORM:
                return "RMS_NORM";
            case GGML_OP_ROPE:
                return "ROPE";
            case GGML_OP_MUL:
                return "MUL";
            case GGML_OP_UNARY:
                {
                    int unary_op = tensor->op_params[0];
                    switch (unary_op) {
                        case GGML_UNARY_OP_SILU:
                            return "UNARY_SILU";
                        case GGML_UNARY_OP_GELU:
                            return "UNARY_GELU";
                        // ... other cases
                        default:
                            return "UNARY_UNKNOWN";
                    }
                }
            case GGML_OP_VIEW:
                return "VIEW";
            case GGML_OP_CPY:
                return "COPY";
            case GGML_OP_CONT:
                return "CONT";
            case GGML_OP_RESHAPE:
                return "RESHAPE";
            default:
                return "UNKNOWN";
        }
    }

    // Recursive tracing with depth for indentation
    void trace_recursive(const ggml_tensor * tensor, int depth) {
        if (!tensor || visited.count(tensor)) {
            return;
        }
        visited.insert(tensor);

        // Print with indentation
        // for (int i = 0; i < depth; ++i) {
        //     std::cout << "  ";
        // }

        // --- Your print_tensor_info logic here, but adapted for this function ---
        debug_tensor(tensor);
        std::cout << std::endl;

        // Recurse on source tensors
        if (tensor->op != GGML_OP_NONE) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i]) {
                    trace_recursive(tensor->src[i], depth + 1);
                }
            }
        }

        // Recurse on view source
        if (tensor->view_src) {
            trace_recursive(tensor->view_src, depth + 1);
        }
    }




  public:
    void trace_tensor_graph(const ggml_tensor * final_tensor, const std::string & name = "OUTPUT") {
        visited.clear();
        std::cout << "\n=== TENSOR GRAPH TRACE: " << name << " ===" << std::endl;
        trace_recursive(final_tensor, 0);
        std::cout << "=== END TRACE ===\n" << std::endl;
    }
};

// Convenience function
void trace_computation_graph(const ggml_tensor * tensor, const std::string & name = "OUTPUT") {
    TensorGraphTracer tracer;
    tracer.trace_tensor_graph(tensor, name);
}
