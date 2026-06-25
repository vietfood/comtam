#include "comtam/core/context.h"
#include "comtam/core/storage.h"
#include "comtam/utils/rng.h"
#include "comtam/utils/common.h"
#include "comtam/core/ops.h"

#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace comtam::core;

constexpr int N = 100;

inline void sanity_add(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

inline void sanity_check(const std::vector<float>& A, const std::vector<float>& B) {
    for (int i = 0; i < N; ++i) {
        if (std::abs(A[i] - B[i]) > 1e-5) {
            throw std::runtime_error("[ERROR] Sanity check failed");
        }
    }
}

int main(int argc, char* argv[]) {
    Context context;
    auto& device = context.device();
    auto& kernels = context.kernels();

    // create array for testing
    std::vector<float> A = comtam::utils::generate_random_array<float>(N, 0, 1);
    std::vector<float> B = comtam::utils::generate_random_array<float>(N, 0, 1);

    // create output array and sanity check
    std::vector<float> C(N);
    std::vector<float> D(N);

    // create buffer
    Storage A_buf = device.allocate(N * sizeof(float));
    Storage B_buf = device.allocate(N * sizeof(float));
    Storage C_buf = device.allocate(N * sizeof(float));

    // copy buffer
    device.copy(A.data(), A.size(), A_buf);
    device.copy(B.data(), B.size(), B_buf);

    // create a command
    Command add_cmd = {Op::ADD, &A_buf, &B_buf, &C_buf, N};

    // finally calculate
    device.submit(add_cmd, kernels);

    // copy back
    device.copy(C_buf, C.data(), C.size());

    // sanity check again
    sanity_add(A, B, D);
    sanity_check(C, D);
    std::cout << "Sanity check passed." << std::endl;

    return 0;
}
