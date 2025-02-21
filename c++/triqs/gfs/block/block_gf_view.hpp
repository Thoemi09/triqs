// Copyright (c) 2020-2023 Simons Foundation
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

#include "block_gf.hpp"
#include "../gf/mpi.hpp"

namespace triqs::gfs {

  template <typename Mesh, typename Target, typename Layout, int Arity, bool IsConst>
  class block_gf_view : is_view_tag, TRIQS_CONCEPT_TAG_NAME(BlockGreenFunction) {
    using this_t = block_gf_view; // for common code

    public:
    static constexpr bool is_view  = true;
    static constexpr bool is_const = IsConst;
    static constexpr int arity     = Arity;

    using mesh_t   = Mesh;
    using target_t = Target;

    using regular_type      = block_gf<Mesh, Target, typename Layout::contiguous_t, Arity>;
    using mutable_view_type = block_gf_view<Mesh, Target, Layout, Arity>;
    using view_type         = block_gf_view<Mesh, Target, Layout, Arity, false>;
    using const_view_type   = block_gf_view<Mesh, Target, Layout, Arity, true>;

    /// The associated real type
    using real_t = block_gf_view<Mesh, typename Target::real_t, Layout, Arity, IsConst>;

    using g_t           = std::conditional_t<IsConst, gf_const_view<Mesh, Target, Layout>, gf_view<Mesh, Target, Layout>>;
    using data_t        = std::conditional_t<Arity == 1, std::vector<g_t>, std::vector<std::vector<g_t>>>;
    using block_names_t = std::conditional_t<Arity == 1, std::vector<std::string>, std::vector<std::vector<std::string>>>;

    std::string name;

    private:
    block_names_t _block_names;
    data_t _glist;

    // ---------------  Constructors --------------------

    struct impl_tag {};
    template <typename G> block_gf_view(impl_tag, G &&x) : name(x.name), _block_names(x.block_names()), _glist(factory<data_t>(x.data())) {}

    public:
    /// Copy constructor
    block_gf_view(block_gf_view const &x) = default;

    /// Move constructor
    block_gf_view(block_gf_view &&) = default;

    /// Construct from block_names and list of gf
    block_gf_view(block_names_t b, data_t d) : _block_names(std::move(b)), _glist(std::move(d)) {
      if constexpr (Arity == 1) {
        if (_glist.size() != _block_names.size())
          TRIQS_RUNTIME_ERROR << "block_gf(vector<string>, vector<gf>) : the two vectors do not have the same size !";
      } else {
        if (_glist.size() != _block_names[0].size())
          TRIQS_RUNTIME_ERROR << "block2_gf(vector<vector<string>>, vector<vector<gf>>) : Outer vectors have different sizes !";
        if (_glist.size() != 0)
          if (_glist[0].size() != _block_names[1].size())
            TRIQS_RUNTIME_ERROR << "block2_gf(vector<vector<string>>, vector<vector<gf>>) : Inner vectors have different sizes !";
      }
    }

    // ---------------  Constructors --------------------

    block_gf_view() = default;

    template <typename L>
    block_gf_view(block_gf<Mesh, Target, L, Arity> const &g)
      requires(IsConst)
       : block_gf_view(impl_tag{}, g) {}

    template <typename L>
    block_gf_view(block_gf<Mesh, Target, L, Arity> &g)
      requires(!IsConst)
       : block_gf_view(impl_tag{}, g) {}

    template <typename L> block_gf_view(block_gf<Mesh, Target, L, Arity> &&g) noexcept : block_gf_view(impl_tag{}, std::move(g)) {}

    template <typename L>
    block_gf_view(block_gf_view<Mesh, Target, L, Arity, !IsConst> const &g)
      requires(IsConst)
       : block_gf_view(impl_tag{}, g) {}

    /// ---------------  Operator = --------------------

    /// Copy the data, without resizing the view.
    block_gf_view &operator=(block_gf_view const &rhs)
      requires(not IsConst)
    {
      _assign_impl(rhs);
      return *this;
    }

    /**
     *  RHS can be anything with .block_names() and [n] -> gf or a scalar
     */
    template <typename RHS>
    block_gf_view &operator=(RHS const &rhs)
      requires(not IsConst)
    {
      if constexpr (not nda::is_scalar_v<RHS>) {
        if (!(size() == rhs.size())) TRIQS_RUNTIME_ERROR << "Gf Assignment in View : incompatible size" << size() << " vs " << rhs.size();
        _assign_impl(rhs);
      } else {
        if constexpr (Arity == 1) {
          for (auto &y : _glist) y = rhs;
        } else {
          for (auto &x : _glist)
            for (auto &y : x) y = rhs;
        }
      }
      return *this;
    }

    /**
    * Assignment operator overload specific for lazy_transform objects
    *
    * @param rhs The lazy object returned e.g. by fourier(my_block_gf)
    */
    template <typename L, typename G>
    block_gf_view &operator=(lazy_transform_t<L, G> const &rhs)
      requires(not IsConst)
    {
      if constexpr (Arity == 1) {
        for (int i = 0; i < rhs.value.size(); ++i) (*this)[i] = rhs.lambda(rhs.value[i]);
      } else {

        for (int i = 0; i < rhs.value.size1(); ++i)
          for (int j = 0; j < rhs.value.size2(); ++j) (*this)(i, j) = rhs.lambda(rhs.value(i, j));
      }
      return *this;
    }

    /**
     * @brief Assignment operator overload for `mpi::lazy` objects.
     *
     * @details It simply calls `mpi::reduce` on each Green's function stored in the lazy object separately and assigns
     * the result to the corresponding Green's function in `this` object.
     *
     * @param l `mpi::lazy` object returned by triqs::gfs::lazy_mpi_reduce.
     * @return Reference to `this` object.
     */
    block_gf_view &operator=(mpi::lazy<mpi::tag::reduce, const_view_type> l)
      requires(not IsConst)
    {
      if constexpr (Arity == 1) {
        if (l.rhs.size() != this->size()) TRIQS_RUNTIME_ERROR << "Error in block_gf_view::operator=: Incompatible sizes";
        for (int i = 0; i < size(); ++i) _glist[i] = triqs::gfs::lazy_mpi_reduce(l.rhs.data()[i], l.c, l.root, l.all, l.op);
      } else {
        if (l.rhs.size1() != this->size1() || l.rhs.size2() != this->size2())
          TRIQS_RUNTIME_ERROR << "Error in block_gf_view::operator=: Incompatible sizes";
        for (int i = 0; i < size1(); ++i)
          for (int j = 0; j < size2(); ++j) _glist[i][j] = triqs::gfs::lazy_mpi_reduce(l.rhs.data()[i][j], l.c, l.root, l.all, l.op);
      }
      _block_names = l.rhs.block_names();
      return *this;
    }

    // ---------------  Rebind --------------------
    /// Rebind
    void rebind(block_gf_view x) noexcept {
      _block_names = x._block_names;
      _glist       = data_t{x._glist}; // copy of vector<vector<gf_view>>, makes new views on the gf of x
      name         = x.name;
    }
    void rebind(block_gf_view<Mesh, Target, Layout, Arity, !IsConst> const &X) noexcept
      requires(IsConst)
    {
      rebind(block_gf_view{X});
    }
    void rebind(block_gf<Mesh, Target, Layout, Arity> const &X) noexcept
      requires(IsConst)
    {
      rebind(block_gf_view{X});
    }
    void rebind(block_gf<Mesh, Target, Layout, Arity> &X) noexcept { rebind(block_gf_view{X}); }

    public:
    //----------------------------- print  -----------------------------
    friend std::ostream &operator<<(std::ostream &out, block_gf_view const &) { return out << "block_gf_view"; }

    // Common code for gf, gf_view, gf_const_view
#include "./_block_gf_view_common.hpp"
  };

} // namespace triqs::gfs

/*------------------------------------------------------------------------------------------------------
 *             Delete std::swap for views
 *-----------------------------------------------------------------------------------------------------*/
namespace std {
  template <typename Mesh, typename Target, typename Layout, int Arity, bool IsConst>
  void swap(triqs::gfs::block_gf_view<Mesh, Target, Layout, Arity, IsConst> &a,
            triqs::gfs::block_gf_view<Mesh, Target, Layout, Arity, IsConst> &b) = delete;
} // namespace std
