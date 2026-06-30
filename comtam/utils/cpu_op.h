#pragma once

#include <cstddef>
#include <limits>

namespace comtam::utils {
  template <typename T>
  inline void add(const T* A, const T* B, T* C, size_t N) {
    for (size_t i = 0; i < N; ++i) {
      C[i] = A[i] + B[i];
    }
  }

  template <typename T>
  inline void sub(const T* A, const T* B, T* C, size_t N) {
    for (size_t i = 0; i < N; ++i) {
      C[i] = A[i] - B[i];
    }
  }

  template <typename T>
  inline void mul(const T* A, const T* B, T* C, size_t N) {
    for (size_t i = 0; i < N; ++i) {
      C[i] = A[i] * B[i];
    }
  }

  template <typename T>
  inline void div(const T* A, const T* B, T* C, size_t N) {
    for (size_t i = 0; i < N; ++i) {
      if (B[i] == 0) {
        C[i] = std::numeric_limits<T>::infinity();
      } else {
        C[i] = A[i] / B[i];
      }
    }
  }
}