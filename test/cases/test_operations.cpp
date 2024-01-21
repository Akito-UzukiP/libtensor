#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <xtensor/xrandom.hpp>
#include <xtensor/xpad.hpp>

#include "benchmark/api.hpp"
#include "../utils/fixture.hpp"
#include "../utils/helper.hpp"
#include "../utils/xt_adaptor.hpp"

TEST_F(TensorTest, Concate) {
    for (int run = 0; run < 5; run++) {
        RANDOM_SHAPE(base_shape)

        auto cat_dim = std::uniform_int_distribution<size_t>(0, base_shape_dim - 1)(rng);
        auto cat_size = std::uniform_int_distribution<size_t>(2, 5)(rng);

        std::vector<xt::xarray<float>> xarrs;
        for (int i = 0; i < cat_size; i++) {
            auto alt_shape = base_shape;
            alt_shape[cat_dim] = shape_dist(rng);
            spdlog::debug("#{} shape: ({})", i, fmt::join(alt_shape, ", "));

            auto xarr = xt::eval(xt::random::randn<float>(alt_shape));
            xarrs.push_back(xarr);
        }

        auto cat_xarr = xarrs[0];
        for (int i = 1; i < cat_size; i++) {
            cat_xarr = xt::concatenate(xt::xtuple(cat_xarr, xarrs[i]), cat_dim);
        }

        spdlog::debug("cat_dim: {}", cat_dim);
        spdlog::debug("cat_shape: ({})", fmt::join(cat_xarr.shape(), ", "));

        std::vector<ts::Tensor<float>> tensors;
        std::transform(xarrs.begin(), xarrs.end(), std::back_inserter(tensors), xtada::to_tensor<float>);
        auto cat_tensor = bm::concat(tensors, cat_dim);

        ASSERT_EQUIV(cat_xarr, cat_tensor);
    }
}

TEST_F(TensorTest, Tile) {
    for (int run = 0; run < 5; run++) {
        RANDOM_SHAPE(base_shape)
        RANDOM_SHAPE_SPEC_DIM(tile_shape, base_shape_dim)

        auto xarr = xt::eval(xt::random::randn<float>(base_shape));
        auto tensor = xtada::to_tensor(xarr);

        spdlog::debug("base_shape: ({})", fmt::join(base_shape, ", "));
        spdlog::debug("tile_shape: ({})", fmt::join(tile_shape, ", "));

        auto xarr_tile = xt::tile(xarr, tile_shape);
        spdlog::debug("xarr tile shape: ({})", fmt::join(xarr_tile.shape(), ", "));

        auto tensor_tile = bm::tile(tensor, tile_shape);

        ASSERT_EQUIV(xarr_tile, tensor_tile);
    }
}

TEST_F(TensorTest, Transpose) {
    for (int run = 0; run < 5; run++) {
        auto dim_dist = std::uniform_int_distribution<size_t>(2, 5);
        RANDOM_SHAPE_SPEC_DIM(base_shape, dim_dist(rng))

        auto xarr = xt::eval(xt::random::randn<float>(base_shape));
        auto tensor = xtada::to_tensor(xarr);
        auto dim1 = std::uniform_int_distribution<size_t>(0, base_shape_dim - 1)(rng);
        auto dim2 = std::uniform_int_distribution<size_t>(0, base_shape_dim - 1)(rng);
        while (dim1 == dim2) {
            dim2 = std::uniform_int_distribution<size_t>(0, base_shape_dim - 1)(rng);
        }
        std::vector<size_t> permute(base_shape_dim);
        std::iota(permute.begin(), permute.end(), 0);
        std::swap(permute[dim1], permute[dim2]);

        auto xarr_transp = xt::transpose(xarr, permute);
        auto tensor_transp = bm::transpose(tensor, dim1, dim2);

        ASSERT_EQUIV(xarr_transp, tensor_transp);
    }
}

TEST_F(TensorTest, Permute) {
    for (int run = 0; run < 5; run++) {
        RANDOM_SHAPE(base_shape)

        auto xarr = xt::eval(xt::random::randn<float>(base_shape));
        auto tensor = xtada::to_tensor(xarr);

        std::vector<size_t> permute(base_shape_dim);
        std::iota(permute.begin(), permute.end(), 0);
        std::shuffle(permute.begin(), permute.end(), rng);

        spdlog::debug("base_shape: ({})", fmt::join(base_shape, ", "));
        spdlog::debug("permute: ({})", fmt::join(permute, ", "));
        spdlog::debug("xarr shape: ({})", fmt::join(xarr.shape(), ", "));

        auto xarr_perm = xt::transpose(xarr, permute);
        spdlog::debug("xarr perm shape: ({})", fmt::join(xarr_perm.shape(), ", "));

        auto tensor_perm = bm::permute(tensor, permute);

        ASSERT_EQUIV(xarr_perm, tensor_perm);
    }
}
