#include "xt_adaptor.hpp"

#include <spdlog/spdlog.h>

namespace xtada {

    template<>
    bool equals(const double &a, const double &b) {
        return std::isnan(a) && std::isnan(b)
               || std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b)
               || std::abs(a - b) < 1e-6;
    }

    template<>
    bool equals(const double &a, const float &b) {
        return equals(a, static_cast<double>(b));
    }

    template<>
    bool equals(const float &a, const double &b) {
        return equals(static_cast<double>(a), b);
    }

    template<>
    bool equals(const float &a, const float &b) {
        return equals(static_cast<double>(a), static_cast<double>(b));
    }
}
