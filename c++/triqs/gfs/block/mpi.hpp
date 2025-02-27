// Copyright (c) 2020-2021 Simons Foundation
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You may obtain a copy of the License at
//     https://www.gnu.org/licenses/gpl-3.0.txt
//
// Authors: Michel Ferrero, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides MPI routines for block Green's function objects.
 */

#pragma once

#include "./block_gf.hpp"
#include "../../utility/exceptions.hpp"

#include <itertools/itertools.hpp>
#include <mpi/mpi.hpp>

#include <cstddef>
#include <numeric>
#include <string>
#include <type_traits>

namespace triqs::gfs {

  namespace detail {

    // Check the shape of block GFs across all processes.
    template <typename G> bool have_mpi_equal_shape(const G &bg, const mpi::communicator &comm) {
      if constexpr (G::arity == 1) {
        return mpi::all_equal(bg.size(), comm);
      } else {
        return mpi::all_equal(bg.size1(), comm) && mpi::all_equal(bg.size2(), comm);
      }
    }

    // Resize a vector of regular GFs or check if the size of a vector of GF views is correct.
    template <bool is_view> void resize_or_check_if_view(auto &g_vec, int size) {
      if constexpr (is_view) {
        if (g_vec.size() != size) TRIQS_RUNTIME_ERROR << "Error in triqs::gfs::detail::resize_or_check_if_view: Vector of GF views has wrong size";
      } else {
        g_vec.resize(size);
      }
    }

    // Hash the block names for comparison.
    template <typename G> std::size_t hash_names(G const &bg) {
      auto view = itertools::transform(bg.block_names(), [](auto const &name) { return std::hash<std::string>{}(name); });
      return std::accumulate(view.begin(), view.end(), std::size_t{0});
    }

  } // namespace detail

  /**
   * @brief Implementation of an MPI broadcast for triqs::gfs::block_gf and triqs::gfs::block_gf_view types.
   *
   * @details It simply broadcasts the vector (of vectors) of GF objects. Furthermore,
   * - for non-view block GFs, it broadcasts the vector (of vectors) of block names and
   * - for views, it expects the block names to be the same on all processes.
   *
   * @tparam G Block GF type.
   * @param bg Block GF (view) to be broadcasted from/into.
   * @param c `mpi::communicator` object.
   * @param root Rank of the root process.
   */
  template <typename G>
    requires(BlockGreenFunction_v<G>)
  void mpi_broadcast(G &&bg, mpi::communicator c, int root) { // NOLINT (temporary views are allowed)
    constexpr bool is_view = std::decay_t<G>::is_view;

    // broadcast block names
    if constexpr (!is_view) {
      // for non-view block GFs, we broadcast the block names directly into the block GF
      mpi::broadcast(bg._block_names, c, root);
    } else {
      // for views, we keep the block names in the block GF but check that they are the same as on the root process
      auto names = bg.block_names();
      mpi::broadcast(names, c, root);
      // should we throw here instead?
      EXPECTS(bg.block_names() == names);
    }

    // broadcast data
    mpi::broadcast(bg.data(), c, root);
  }

  /**
   * @brief Implementation of an MPI reduce for triqs::gfs::block_gf and triqs::gfs::block_gf_view types using a C-style
   * API.
   *
   * @details The function reduces input Block GFs (views) from all processes in the given communicator and makes the
   * result available on the root process (`all == false`) or on all processes (`all == true`).
   *
   * It throws an exception if the input block GFs on all processes and the output block GF views on receiving processes
   * do not have the same shape.
   *
   * The block names of the input block GFs on all processes and of output block GF views on receiving processes are
   * expected to be the same.
   *
   * The content of the output block GF depends on the MPI rank and whether it receives the data or not:
   * - On receiving ranks, it contains the reduced GF objects obtained by calling triqs::gfs::mpi_reduce_capi on each
   * GF separately and the same block names as the input block GF (the block names are assigned for non-views but for
   * views, it is expected that they already have the correct block names).
   * - On non-receiving ranks, the output block GF is ignored and left unchanged.
   *
   * @tparam G1 Block GF type.
   * @tparam G2 Block GF type.
   * @param bg_in Block GF (view) to be reduced.
   * @param bg_out Block GF (view) to be reduced into.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op MPI reduction operation.
   */
  template <typename G1, typename G2>
    requires(BlockGreenFunction_v<G1> and BlockGreenFunction_v<G2>)
  void mpi_reduce_capi(G1 const &bg_in, G2 &&bg_out, mpi::communicator c, int root, // NOLINT (temporary views are allowed here)
                       bool all, MPI_Op op) {
    constexpr bool is_view = std::decay_t<G2>::is_view;
    bool const receive     = (c.rank() == root || all);

    // check the shape and block names of the input block GFs
    EXPECTS(mpi::all_equal(detail::hash_names(bg_in), c));
    // can we remove this check? shape should be equal if the block names are equal
    if (not detail::have_mpi_equal_shape(bg_in, c))
      TRIQS_RUNTIME_ERROR << "Error in triqs::gfs::mpi_reduce_capi: Shapes of input block GFs must be equal";

    // assign (check) the block names of the output block GF (view) on receiving ranks
    if (receive) {
      if constexpr (is_view) {
        EXPECTS(detail::hash_names(bg_in) == detail::hash_names(bg_out));
      } else {
        bg_out._block_names = bg_in.block_names();
      }
    }

    // dummy GF for non-receiving ranks
    auto g_dummy = typename std::decay_t<G2>::g_t::regular_type{};

    // reduce each GF separately
    if constexpr (G1::arity == 1) {
      if (receive) detail::resize_or_check_if_view<is_view>(bg_out.data(), bg_in.size());
      for (int i = 0; i < bg_in.size(); ++i) mpi_reduce_capi(bg_in.data()[i], receive ? bg_out.data()[i] : g_dummy, c, root, all, op);
    } else {
      if (receive) detail::resize_or_check_if_view<is_view>(bg_out.data(), bg_in.size1());
      for (int i = 0; i < bg_in.size1(); ++i) {
        if (receive) detail::resize_or_check_if_view<is_view>(bg_out.data()[i], bg_in.size2());
        for (int j = 0; j < bg_in.size2(); ++j) mpi_reduce_capi(bg_in.data()[i][j], receive ? bg_out.data()[i][j] : g_dummy, c, root, all, op);
      }
    }
  }

  /**
   * @brief Implementation of an in-place MPI reduce for triqs::gfs::block_gf and triqs::gfs::block_gf_view types.
   *
   * @details The function in-place reduces input block GFs (views) from all processes in the given communicator and
   * makes the result available on the root process (`all == false`) or on all processes (`all == true`).
   *
   * It simply calls `mpi::reduce_in_place` on the data of the block GF, i.e. on vector (of vectors) of GF objects.
   *
   * The block GFs are expected to have the same block names on all processes.
   *
   * @tparam G Block GF type.
   * @param bg Block GF (view) to be reduced (into).
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op MPI reduction operation.
   */
  template <typename G>
    requires(BlockGreenFunction_v<G>)
  void mpi_reduce_in_place(G &&bg, mpi::communicator c = {}, int root = 0, bool all = false, // NOLINT (temporary views are allowed)
                           MPI_Op op = MPI_SUM) {
    EXPECTS(mpi::all_equal(detail::hash_names(bg), c));
    mpi::reduce_in_place(bg.data(), c, root, all, op);
  }

  /**
   * @brief Implementation of an MPI reduce for triqs::gfs::block_gf and triqs::gfs::block_gf_view types.
   *
   * @details The function reduces input block GFs (views) from all processes in the given communicator and makes the
   * result available on the root process (`all == false`) or on all processes (`all == true`).
   *
   * The block GFs are expected to have the same shape and block names on all processes.
   *
   * The content of the returned block GF depends on the MPI rank and whether it receives the data or not:
   * - On receiving ranks, it contains the reduced GF objects and the same block names as the input GF.
   * - On non-receiving ranks, a default constructed block GF is returned.
   *
   * @tparam G Block GF type.
   * @param bg Block GF (view) to be reduced (into).
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op MPI reduction operation.
   * @return A triqs::gfs::block_gf object with the reduced data.
   */
  template <typename G>
    requires(BlockGreenFunction_v<G>)
  auto mpi_reduce(G const &bg, mpi::communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    auto res = typename G::regular_type{};
    mpi_reduce_capi(bg, res, c, root, all, op);
    return res;
  }

  /**
   * @brief Implementation of a lazy MPI reduce for triqs::gfs::block_gf and triqs::gfs::block_gf_view types.
   *
   * @details This function is lazy, i.e. it returns an `mpi::lazy<mpi::tag::reduce, G::const_view_type>` object without
   * performing the actual MPI operation. However, the returned object can be used to initialize/assign to
   * triqs::gfs::block_gf and triqs::gfs::block_gf_view types.
   *
   * See also triqs::gfs::block_gf::operator=(mpi::lazy<mpi::tag::reduce, block_gf::const_view_type>) and
   * triqs::gfs::block_gf::operator=(mpi::lazy<mpi::tag::reduce, block_gf::const_view_type>).
   *
   * @tparam G Block GF type.
   * @param bg Block GF (view) to be reduced.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op MPI reduction operation.
   * @return An `mpi::lazy<mpi::tag::reduce, G::const_view_type>` object.
   */
  template <typename G>
    requires(BlockGreenFunction_v<G>)
  auto lazy_mpi_reduce(G const &bg, mpi::communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    return mpi::lazy<mpi::tag::reduce, typename G::const_view_type>{bg(), c, root, all, op};
  }

} // namespace triqs::gfs
