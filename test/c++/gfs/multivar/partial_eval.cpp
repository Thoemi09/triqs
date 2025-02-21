// Copyright (c) 2017-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2017-2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2022 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

#include <triqs/test_tools/gfs.hpp>

using namespace std::complex_literals;

TEST(Gf, PartialEval) {

  double beta   = 1.;
  double tmin   = 0.;
  double tmax   = 1.0;
  int n_re_time = 10;
  int n_im_time = 10;
  double wmin   = 0.;
  double wmax   = 1.0;
  int n_re_freq = 10;
  int n_im_freq = 10;

  nda::clef::placeholder<0> w_;
  nda::clef::placeholder<1> wn_;
  nda::clef::placeholder<2> tau_;

  auto G_w     = gf<refreq, scalar_valued>{{wmin, wmax, n_re_freq}};
  auto G_t_tau = gf<prod<retime, imtime>, scalar_valued>{{{tmin, tmax, n_re_time}, {beta, Fermion, n_im_time}}};
  auto G_w_wn  = gf<prod<refreq, imfreq>, scalar_valued>{{{wmin, wmax, n_re_freq}, {beta, Fermion, n_im_freq}}};
  auto G_w_tau = gf<prod<refreq, imtime>, scalar_valued>{{{wmin, wmax, n_re_freq}, {beta, Fermion, n_im_time}}};

  G_w_wn(w_, wn_) << 1 / (wn_ - 1) / (w_ * w_ * w_ + 1i);
  G_w_tau(w_, tau_) << exp(-2 * tau_) / (w_ * w_ + 1);

  int index  = n_re_freq / 3;
  double tau = std::get<1>(G_w_tau.mesh().components())[index];

  G_w(w_) << exp(-2 * tau) / (w_ * w_ + 1);

  auto G_w_wn_sl0_a = G_w_wn[8, all_t()];
  auto G_w_wn_sl0_b = G_w_wn[all_t(), 3];

  static_assert(std::is_same<std::remove_reference_t<decltype(G_w_wn_sl0_a.mesh())>, const mesh::imfreq>::value, "oops");

  EXPECT_CLOSE((G_w_wn[8, 3]), G_w_wn_sl0_a[3]);
  EXPECT_CLOSE((G_w_wn[8, 3]), G_w_wn_sl0_b[8]);

  rw_h5(G_t_tau, "G_t_tau");
  rw_h5(G_w_wn, "G_w_wn");
  rw_h5(G_w_tau, "G_w_tau");
}

MAKE_MAIN;
