// Copyright (c) 2016-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2016-2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2020 Simons Foundation
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

namespace triqs::gfs {

  /*------------------------------------------------------------------------------------------------------
  *                              HDF5
  *-----------------------------------------------------------------------------------------------------*/

  template <typename G, typename Target> constexpr bool gf_has_target() { return std::is_same<typename G::target_t, Target>::value; }
  /*
 // ---------------------------

 // Some work that may be necessary before writing (some compression, see imfreq)
 // Default : do nothing
 template <typename M, typename T> struct gf_h5_before_write {
  template <typename G> static G const &invoke(h5::group gr, G const &g) { return g; }
 };

 // Before writing to h5, check if I can save the positive freq only
 template <typename T> struct gf_h5_before_write<imfreq, T> {
  template <typename G> static gf_const_view<imfreq, T> invoke(h5::group gr, G const &g) {
   if (is_gf_real_in_tau(g, 1.e-13)) return positive_freq_view(g);
   return g;
  }
 };
*/
  // ---------------------------

  // FIXME : C17 : REMOVE THIS dispatch with a constexpr if
  // Some work that may be necessary after the read (for backward compatibility e.g.)
  // Default : do nothing
  template <typename M, typename T> struct gf_h5_after_read {
    template <typename G> static void invoke(h5::group, G &) {}
  };

  // After reading from h5, is the function is for freq >0, unfold it to the full mesh
  template <typename T> struct gf_h5_after_read<mesh::imfreq, T> {
    template <typename G> static void invoke(h5::group, G &g) {
      if (g.mesh().positive_only()) g = make_gf_from_real_gf(make_const_view(g));
    }
  };
  // same, for python interface
  template <typename T> gf<mesh::imfreq, T> _gf_h5_after_read(gf_view<mesh::imfreq, T> g) {
    if (g.mesh().positive_only())
      return make_gf_from_real_gf(make_const_view(g));
    else
      return g;
  }

  // ---------------------------

  // the h5 write and read of gf members, so that we can specialize it e.g. for block gf
  template <typename V, typename T> struct gf_h5_rw {

    //template <typename G> static void write(h5::group gr, G const &g) { _write(gr, gf_h5_before_write<V, T>::invoke(gr, g)); }

    template <typename G> static void write(h5::group gr, G const &g) {
      // write the data
      //constexpr bool _can_compress = (gf_has_target<G, imtime>() or gf_has_target<G, legendre>());
      //if (_can_compress and is_gf_real(g))
      // h5_write(gr, "data", array<double, G::data_t::rank>(real(g.data())));
      //else
      h5_write(gr, "data", g.data());
      h5_write(gr, "mesh", g._mesh);
    }

    template <typename G> static void read(h5::group gr, G &g) {
      h5_read(gr, "data", g._data);
      h5_read(gr, "mesh", g._mesh);
      gf_h5_after_read<V, T>::invoke(gr, g);
    }
  };

} // namespace triqs::gfs
