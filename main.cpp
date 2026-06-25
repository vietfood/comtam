#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

#include "comtam/core/context.h"
#include "comtam/utils/rng.h"

#include "comtam/tensor.h"

using namespace comtam;

inline void sanity_check(const std::vector<float>& A, const std::vector<float>& B) {
    for (int i = 0; i < A.size(); ++i) {
        if (std::abs(A[i] - B[i]) > 1e-5) {
            throw std::runtime_error("[ERROR] Sanity check failed");
        }
    }
}

constexpr int N = 20;

int main(int argc, char* argv[]) {
    core::Context context;
    auto& device = context.device();
    auto& kernels = context.kernels();

    // create array for testing
    std::vector<float> A = comtam::utils::generate_random_array<float>(N, 0, 1);
    std::vector<float> B = comtam::utils::generate_random_array<float>(N, 0, 1);

    // test to vector
    Tensor tA(A.data(), {N}, device);
    auto C = tA.to_vector(device);
    sanity_check(A, C);

    // test from vector
    Tensor tB({N}, device);
    tB.from_vector(B, device);
    auto D = tB.to_vector(device);
    sanity_check(B, D);

    std::cout << "Sanity check passed" << std::endl;

    return 0;
}
