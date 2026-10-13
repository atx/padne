#pragma once

#include <initializer_list>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// Hands the vector's buffer over to a numpy array of the given shape without
// copying.
template <typename T>
nanobind::ndarray<nanobind::numpy, T> vector_to_numpy(std::vector<T> &&v,
                                                      std::initializer_list<size_t> shape)
{
    auto *buf = new std::vector<T>(std::move(v));
    nanobind::capsule owner(buf, [](void *p) noexcept {
        delete static_cast<std::vector<T> *>(p);
    });
    return nanobind::ndarray<nanobind::numpy, T>(buf->data(), shape, owner);
}
