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

#include "./gf.hpp"

#include <mpi/mpi.hpp>

namespace triqs::gfs {

  /**
   * @brief Implementation of an MPI broadcast for triqs::gfs::gf, triqs::gfs::gf_view or triqs::gfs::gf_const_view
   * types.
   *
   * @details It simply broadcasts the data of the GF, i.e. the nda array/view object.
   *
   * @note It does not broadcast the mesh or check if the mesh is the same on all processes.
   *
   * @tparam G triqs::gfs::MemoryGf type.
   * @param g GF (view) to be broadcasted from/into.
   * @param c `mpi::communicator` object.
   * @param root Rank of the root process.
   */
  template <MemoryGf G> void mpi_broadcast(G &&g, mpi::communicator c, int root) { // NOLINT (temporary views are allowed)
    mpi::broadcast(g._mesh, c, root);
    mpi::broadcast(g.data(), c, root);
  }

  /**
   * @brief Implementation of an in-place MPI reduce for triqs::gfs::gf, triqs::gfs::gf_view or
   * triqs::gfs::gf_const_view types.
   *
   * @details The function in-place reduces input GFs (views) from all processes in the given communicator and makes the
   * result available on the root process (`all == false`) or on all processes (`all == true`).
   *
   * It simply calls `mpi::reduce_in_place` on the data of the GF, i.e. the nda array/view object.
   *
   * The meshes are expected to be the same on all processes.
   *
   * @tparam G triqs::gfs::MemoryGf type.
   * @param g GF (view) to be reduced (into).
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op MPI reduction operation.
   */
  template <MemoryGf G>
  void mpi_reduce_in_place(G &&g, mpi::communicator c = {}, int root = 0, bool all = false, // NOLINT (temporary views are allowed)
                           MPI_Op op = MPI_SUM) {
    EXPECTS(mpi::all_equal(g.mesh().mesh_hash(), c));
    mpi::reduce_in_place(g.data(), c, root, all, op);
  }

  /**
   * @brief Implementation of an MPI reduce for triqs::gfs::gf, triqs::gfs::gf_view or triqs::gfs::gf_const_view types.
   *
   * @details The function reduces input GF (views) from all processes in the given communicator and makes the result
   * available on the root process (`all == false`) or on all processes (`all == true`).
   *
   * On receiving processes, the function returns a new GF object with the reduced data. On non-receiving processes, it
   * returns a default constructed GF object.
   *
   * The meshes are expected to be the same on all processes.
   *
   * @tparam G triqs::gfs::MemoryGf type.
   * @param g GF (view) to be reduced.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op MPI reduction operation.
   * @return A triqs::gfs::gf object with the reduced data.
   */
  template <MemoryGf G> auto mpi_reduce(G const &g, mpi::communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    EXPECTS(mpi::all_equal(g.mesh().mesh_hash(), c));
    using return_t = typename G::regular_type;
    auto res       = return_t{};
    if (c.rank() == root || all) res = return_t{g.mesh(), g.target_shape()};
    nda::mpi_reduce_capi(g.data(), res.data(), c, root, all, op);
    return res;
  }

  /**
   * @brief Implementation of a lazy MPI reduce for triqs::gfs::gf, triqs::gfs::gf_view or triqs::gfs::gf_const_view
   * types.
   *
   * @details This function is lazy, i.e. it returns an `mpi::lazy<mpi::tag::reduce, G::const_view_type>` object without
   * performing the actual MPI operation. However, the returned object can be used to initialize/assign to
   * triqs::gfs::gf and triqs::gfs::gf_view objects.
   *
   * See also triqs::gfs::gf::operator=(mpi::lazy<mpi::tag::reduce, gf_const_view<M, Target>>) and
   * triqs::gfs::gf_view::operator=(mpi::lazy<mpi::tag::reduce, gf_const_view<M, Target>>).
   *
   * @tparam G triqs::gfs::MemoryGf type.
   * @param g GF (view) to be reduced.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op MPI reduction operation.
   * @return An `mpi::lazy<mpi::tag::reduce, G::const_view_type>` object.
   */
  template <MemoryGf G> auto lazy_mpi_reduce(G const &g, mpi::communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    return mpi::lazy<mpi::tag::reduce, typename G::const_view_type>{g(), c, root, all, op};
  }

} // namespace triqs::gfs
