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

#pragma once

#include "./block_gf.hpp"
#include "../../utility/exceptions.hpp"

#include <mpi/mpi.hpp>

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

  } // namespace detail

  /**
   * @brief Implementation of an MPI broadcast for triqs::gfs::block_gf and triqs::gfs::block_gf_view types.
   *
   * @details For triqs::gfs::block_gf objects, it broadcasts the whole vector (of vectors) of GFs. For
   * triqs::gfs::block_gf_view objects, it broadcasts each GF separately.
   *
   * It throws an exception if the sizes of triqs::gfs::block_gf_view objects are not equal on all processes.
   *
   * @note It only broadcasts the data of the block GFs. It does not check if the mesh or the block names are the same
   * on all processes.
   *
   * @tparam G Block GF type.
   * @param bg Block GF (view) to be broadcasted from/into.
   * @param c `mpi::communicator` object.
   * @param root Rank of the root process.
   */
  template <typename G>
    requires(BlockGreenFunction_v<G>)
  void mpi_broadcast(G &&bg, mpi::communicator c = {}, int root = 0) { // NOLINT (temporary views are allowed)
    // vector broadcast performs a resize, which does not make sense for views
    if constexpr (std::decay_t<G>::is_view) {
      // for views, broadcast each GF separately
      if (not detail::have_mpi_equal_shape(bg, c)) TRIQS_RUNTIME_ERROR << "Error in triqs::gfs::mpi_broadcast: block_gf_view sizes are not equal.";
      if constexpr (G::arity == 1) {
        for (auto &g : bg.data()) mpi::broadcast(g, c, root);
      } else {
        for (auto &vec : bg.data()) {
          for (auto &g : vec) mpi::broadcast(g, c, root);
        }
      }
    } else {
      // for non-views, we can broadcast the whole vector (of vectors) of GFs
      mpi::broadcast(bg.data(), c, root);
    }
  }

  /**
   * @brief Implementation of an in-place MPI reduce for triqs::gfs::block_gf and triqs::gfs::block_gf_view types.
   *
   * @details The function in-place reduces input block GFs (views) from all processes in the given communicator and
   * makes the result available on the root process (`all == false`) or on all processes (`all == true`).
   *
   * It simply calls `mpi::reduce_in_place` on the data of the block GF.
   *
   * @note It does not check if the mesh or the block names are the same on all processes.
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
    mpi::reduce_in_place(bg.data(), c, root, all, op);
  }

  /**
   * @brief Implementation of an MPI reduce for triqs::gfs::block_gf and triqs::gfs::block_gf_view types.
   *
   * @details The function reduces input block GFs (views) from all processes in the given communicator and makes the
   * result available on the root process (`all == false`) or on all processes (`all == true`).
   *
   * On receiving processes, the function returns a new block GF object with the reduced data. On non-receiving
   * processes, it returns a default constructed block GF object.
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
    // reduce only the data
    auto data = typename G::regular_type::data_t{};
    if constexpr (G::is_view) {
      // for views, reduce each GF separately
      if (not detail::have_mpi_equal_shape(bg, c)) TRIQS_RUNTIME_ERROR << "Error in triqs::gfs::mpi_reduce: block_gf_view shapes are not equal.";
      if constexpr (G::arity == 1) {
        data.reserve(bg.size());
        for (auto const &g : bg.data()) data.emplace_back(mpi::reduce(g, c, root, all, op));
      } else {
        data.resize(bg.size1());
        for (int i = 0; i < bg.size1(); ++i) {
          for (auto const &g : bg.data()[i]) data[i].emplace_back(mpi::reduce(g, c, root, all, op));
        }
      }
    } else {
      // for non-views, we can reduce the whole vector (of vectors) of GFs
      data = mpi::reduce(bg.data(), c, root, all, op);
    }

    // create a new block GF with the reduced data (on receiving ranks)
    if (c.rank() == root || all) return typename G::regular_type{bg.block_names(), data};
    return typename G::regular_type{};
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
