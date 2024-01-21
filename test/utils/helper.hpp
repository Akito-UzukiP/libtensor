#pragma once

#define RANDOM_SHAPE_SPEC_DIM(var_name, dim) \
    std::vector<size_t> var_name; \
    const auto var_name##_dim = dim; \
    for (size_t i = 0; i < var_name##_dim; i++) { \
        var_name.push_back(shape_dist(rng)); \
    }

#define RANDOM_SHAPE(var_name) \
    RANDOM_SHAPE_SPEC_DIM(var_name, dim_dist(rng))

#define EXPECT_EQUIV(expected, actual) \
    EXPECT_TRUE(xtada::equiv(xt::eval(expected), actual))

#define ASSERT_EQUIV(expected, actual) \
    ASSERT_TRUE(xtada::equiv(xt::eval(expected), actual))
