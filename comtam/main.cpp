#include "comtam/core/context.h"
#include "comtam/core/storage.h"
#include "comtam/utils/rng.h"
#include "comtam/utils/common.h"

#include <type_traits>
#include <vector>

using namespace comtam::core;

constexpr int N = 20;

int main(int argc, char* argv[]) {
    // Metal context
    Context context;
    auto& device = context.device();
    auto& kernels = context.kernels();

    // Create array for testing
    std::vector<float> A = comtam::utils::generate_random_array<float>(N, 0, 1);
    std::vector<float> B = comtam::utils::generate_random_array<float>(N, 0, 1);
    comtam::utils::print_array(A.data(), N, "A");
    comtam::utils::print_array(B.data(), N, "B");

    // Create output array
    std::vector<float> C(N);

    // Create buffer
    Storage A_buf = device.allocate(N * sizeof(float));
    Storage B_buf = device.allocate(N * sizeof(float));
    Storage C_buf = device.allocate(N * sizeof(float));

    // Copy buffer
    device.copy(A, A_buf);
    device.copy(B, B_buf);

    // Sanity check
    A_buf.print("A buffer");
    B_buf.print("B buffer");

    return 0;
}
