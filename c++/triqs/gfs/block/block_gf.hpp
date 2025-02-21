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

#include "./../../utility/factory.hpp"
#include "./gf_struct.hpp"
#include "../gf/gf.hpp"
#include "../gf/targets.hpp"

namespace triqs::gfs {

  using triqs::utility::factory;

  /*----------------------------------------------------------
   *   Declaration of main types : gf, gf_view, gf_const_view
   *--------------------------------------------------------*/
  template <typename Mesh, typename Target = matrix_valued, typename Layout = nda::C_layout, int Arity = 1> class block_gf;
  template <typename Mesh, typename Target = matrix_valued, typename Layout = nda::C_stride_layout, int Arity = 1, bool IsConst = false>
  class block_gf_view;

  // aliases
  template <typename Mesh, typename Target = matrix_valued, typename Layout = nda::C_stride_layout, int Arity = 1>
  using block_gf_const_view = block_gf_view<Mesh, Target, Layout, Arity, true>;

  template <typename Mesh, typename Target = matrix_valued, typename Layout = nda::C_layout> using block2_gf = block_gf<Mesh, Target, Layout, 2>;
  template <typename Mesh, typename Target = matrix_valued, typename Layout = nda::C_stride_layout>
  using block2_gf_view = block_gf_view<Mesh, Target, Layout, 2, false>;
  template <typename Mesh, typename Target = matrix_valued, typename Layout = nda::C_stride_layout>
  using block2_gf_const_view = block_gf_view<Mesh, Target, Layout, 2, true>;

  /// --------------------------- CTAD ---------------------------------

  template <typename Mesh, typename Target, typename Layout> block_gf(std::vector<gf<Mesh, Target, Layout>>) -> block_gf<Mesh, Target, Layout, 1>;
  template <typename Mesh> block_gf(Mesh const &, gf_struct_t const &) -> block_gf<Mesh, matrix_valued>;
  template <typename Mesh, typename Target, typename Layout, int Arity, bool IsConst>
  block_gf(block_gf_view<Mesh, Target, Layout, Arity, IsConst>) -> block_gf<Mesh, Target, typename Layout::contiguous_t, Arity>;

  /// ---------------------------  traits ---------------------------------

  // Is G a block_gf, block_gf_view, block_gf_const_view
  // is_block_gf<G> is true iif G is a block_gf or block2_gf
  // is_block_gf<G,1> is true iff G is a block_gf
  // is_block_gf<G,2> is true iff G is a block2_gf
  //
  template <typename G, int n = 0> inline constexpr bool is_block_gf_v = false;

  template <typename Mesh, typename Target, typename Layout, int Arity>
  inline constexpr bool is_block_gf_v<block_gf<Mesh, Target, Layout, Arity>, Arity> = true;

  template <typename Mesh, typename Target, typename Layout, int Arity, bool IsConst>
  inline constexpr bool is_block_gf_v<block_gf_view<Mesh, Target, Layout, Arity, IsConst>, Arity> = true;

  template <typename, typename = std::void_t<>> inline constexpr int arity_of = -1;

  template <typename T> inline constexpr int arity_of<T, std::void_t<decltype(T::arity)>> = T::arity;

  template <typename G> inline constexpr bool is_block_gf_v<G, 0> = is_block_gf_v<G, 1> or is_block_gf_v<G, 2>;

  // Given a gf G, the corresponding block
  template <typename G> using get_mesh_t              = typename std::decay_t<G>::mesh_t;
  template <typename G> using get_target_t            = typename std::decay_t<G>::target_t;
  template <typename G> using block_gf_of             = block_gf<get_mesh_t<G>, get_target_t<G>>;
  template <typename G> using block_gf_view_of        = block_gf_view<get_mesh_t<G>, get_target_t<G>>;
  template <typename G> using block_gf_const_view_of  = block_gf_const_view<get_mesh_t<G>, get_target_t<G>>;
  template <typename G> using block2_gf_of            = block2_gf<get_mesh_t<G>, get_target_t<G>>;
  template <typename G> using block2_gf_view_of       = block2_gf_view<get_mesh_t<G>, get_target_t<G>>;
  template <typename G> using block2_gf_const_view_of = block2_gf_const_view<get_mesh_t<G>, get_target_t<G>>;

  // The trait that "marks" the Green function
  TRIQS_DEFINE_CONCEPT_AND_ASSOCIATED_TRAIT(BlockGreenFunction);

  // ------------- Helper Types -----------------------------

  template <typename Lambda, typename T> struct lazy_transform_t {
    Lambda lambda;
    T value;
  };

  template <typename Lambda, typename T> lazy_transform_t<Lambda, T> make_lazy_transform(Lambda &&l, T &&x) {
    return {std::forward<Lambda>(l), std::forward<T>(x)};
  }

  /// ---------------------------  details  ---------------------------------

  namespace details {
    inline auto _make_block_names1(int n) {
      std::vector<std::string> r(n);
      for (int i = 0; i < n; ++i) r[i] = std::to_string(i);
      return r;
    }
    inline std::vector<std::vector<std::string>> _make_block_names2(int n, int p) { return {_make_block_names1(n), _make_block_names1(p)}; }
  } // namespace details

  /// ---------------------------  implementation  ---------------------------------

  template <typename Mesh, typename Target, typename Layout, int Arity> class block_gf : TRIQS_CONCEPT_TAG_NAME(BlockGreenFunction) {
    using this_t = block_gf; // for common code
    public:
    static constexpr bool is_view  = false;
    static constexpr bool is_const = false;
    static constexpr int arity     = Arity;

    using mesh_t   = Mesh;
    using target_t = Target;

    using regular_type      = block_gf<Mesh, Target, Layout, Arity>;
    using mutable_view_type = block_gf_view<Mesh, Target, typename Layout::with_lowest_guarantee_t, Arity>;
    using view_type         = block_gf_view<Mesh, Target, typename Layout::with_lowest_guarantee_t, Arity, false>;
    using const_view_type   = block_gf_view<Mesh, Target, typename Layout::with_lowest_guarantee_t, Arity, true>;

    /// The associated real type
    using real_t = block_gf<Mesh, typename Target::real_t, Layout, Arity>;

    using g_t           = gf<Mesh, Target, Layout>;
    using data_t        = std::conditional_t<Arity == 1, std::vector<g_t>, std::vector<std::vector<g_t>>>;
    using block_names_t = std::conditional_t<Arity == 1, std::vector<std::string>, std::vector<std::vector<std::string>>>;

    std::string name;

    private:
    block_names_t _block_names;
    data_t _glist;

    // ---------------  Constructors --------------------

    struct impl_tag {};
    template <typename G> block_gf(impl_tag, G &&x) : name(x.name), _block_names(x.block_names()), _glist(factory<data_t>(x.data())) {}

    public:
    /// Copy constructor
    block_gf(block_gf const &x) = default;

    /// Move constructor
    block_gf(block_gf &&) = default;

    /// Construct from block_names and list of gf
    block_gf(block_names_t b, data_t d) : _block_names(std::move(b)), _glist(std::move(d)) {
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

    /// Construct an empty Green function (with empty array).
    block_gf() = default;

    /// From a block_gf_view of the same kind
    template <typename L, bool Cnst> block_gf(block_gf_view<Mesh, Target, L, Arity, Cnst> const &g) : block_gf(impl_tag{}, g) {}

    /// Construct from anything which models BlockGreenFunction.
    // TODO: We would like to refine this, G should have the same mesh, target, at least ...
    template <typename G>
    block_gf(G const &x)
      requires(BlockGreenFunction<G>::value)
       : block_gf() {
      static_assert(G::arity == Arity, "Impossible");
      *this = x;
    }

    /// Construct from the mpi lazy class of the implementation class, cf mpi section
    // NB : type must be the same, e.g. g2(reduce(g1)) will work only if mesh, Target, Singularity are the same...
    template <typename Tag> block_gf(mpi::lazy<Tag, block_gf_const_view<Mesh, Target>> x) : block_gf() { operator=(x); }

    /// Construct from a vector of gf
    block_gf(data_t V)
      requires(Arity == 1)
       : _block_names(details::_make_block_names1(V.size())), _glist(std::move(V)) {}

    /// Constructs a n block
    block_gf(int n)
      requires(Arity == 1)
       : block_gf(data_t(n)) {}

    /// Constructs a n block with copies of g.
    block_gf(int n, g_t const &g)
      requires(Arity == 1)
       : block_gf(data_t(n, g)) {}

    /// Construct from the vector of names and one gf to be copied
    block_gf(block_names_t b, g_t const &g)
      requires(Arity == 1)
       : _block_names(std::move(b)), _glist(_block_names.size(), g) {}

    /// Construct from the vector of names
    block_gf(block_names_t b)
      requires(Arity == 1)
       : _block_names(std::move(b)), _glist(_block_names.size()) {}

    // Create Block Green function from Mesh and gf_struct
    block_gf(Mesh const &m, gf_struct_t const &gf_struct)
      requires(Arity == 1)
    {

      for (auto const &[bl_name, bl_size] : gf_struct) {
        _block_names.push_back(bl_name);
        if constexpr (Target::rank == 0)
          _glist.emplace_back(m);
        else
          _glist.emplace_back(m, make_shape(bl_size, bl_size));
      }
    }

    // Create Block Green function from Mesh and gf_struct
    template <typename Int>
    block_gf(Mesh const &m, std::vector<Int> const &bl_sizes)
      requires(Arity == 1 && std::is_integral_v<Int>)
    {

      for (auto const &[bl, bl_size] : itertools::enumerate(bl_sizes)) {
        _block_names.push_back(std::to_string(bl));
        if constexpr (Target::rank == 0)
          _glist.emplace_back(m);
        else
          _glist.emplace_back(m, make_shape(bl_size, bl_size));
      }
    }

    /// Constructs a n blocks with copies of g.
    block_gf(int n, int p, g_t const &g)
      requires(Arity == 2)
       : _block_names(details::_make_block_names2(n, p)), _glist(n, std::vector<g_t>(p, g)) {}

    /// Construct from a vector of gf
    block_gf(data_t V)
      requires(Arity == 2)
       : _block_names(details::_make_block_names2(V.size(), V[0].size())), _glist(std::move(V)) {}

    /// Construct from an `mpi::lazy` object.
    block_gf(mpi::lazy<mpi::tag::reduce, block_gf::const_view_type> l) : block_gf() { operator=(l); }

    /// ---------------  Operator = --------------------

    block_gf &operator=(block_gf const &rhs) = default;
    block_gf &operator=(block_gf &&rhs)      = default;

    /**
     * @brief Assignment operator overload for `mpi::lazy` objects.
     *
     * @details It simply calls `mpi::reduce` on the block Green's function stored in the lazy object and assigns the
     * result to `this` object.
     *
     * @param l `mpi::lazy` object returned by triqs::gfs::lazy_mpi_reduce.
     * @return Reference to `this` object.
     */
    block_gf &operator=(mpi::lazy<mpi::tag::reduce, block_gf::const_view_type> l) {
      *this = mpi::reduce(l.rhs, l.c, l.root, l.all, l.op);
      return *this;
    }

    /**
     * Assignment operator
     *
     * @tparam RHS Type of the right hand side rhs
     *
     *             RHS can be anything modeling the gf concept TBW
     *             In particular lazy expression with Green functions
     * @param rhs
     *
     * The assignment resizes the mesh and the data, invalidating all pointers on them.
     */
    template <typename RHS> block_gf &operator=(RHS &&rhs) {
      if constexpr (Arity == 1) {
        _glist.resize(rhs.size());
        _block_names.resize(rhs.size());
        _assign_impl(rhs);
      } else {
        _block_names.resize(2);
        _glist.resize(rhs.size1());
        for (auto &g_bl : _glist) g_bl.resize(rhs.size2());
        _block_names[0].resize(rhs.size1());
        _block_names[1].resize(rhs.size2());
        _assign_impl(rhs);
      }
      return *this;
    }

    public:
    //----------------------------- print  -----------------------------

    friend std::ostream &operator<<(std::ostream &out, block_gf const &) { return out << "block_gf"; }

    // Common code for gf, gf_view, gf_const_view
#include "./_block_gf_view_common.hpp"
  };

} // namespace triqs::gfs
