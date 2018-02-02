/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2012-2016 by M. Ferrero, O. Parcollet
 * Copyright (C) 2018 The Simons Foundation, Authors: H. UR Strand, M. Zingl
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once

namespace triqs {
  namespace gfs {

    //-------------------------------------------------------
    // For Imaginary Matsubara Frequency functions
    // ------------------------------------------------------

    /// Density
    /**
     * Computes the density of the Gf g, i.e $g(\tau=0^-)$
     * Uses tail moments n=1, 2, and 3
     */
    arrays::matrix<dcomplex> density(gf_const_view<imfreq> g, array_view<dcomplex, 3> = {});
    dcomplex density(gf_const_view<imfreq, scalar_valued> g, array_view<dcomplex, 1> = {});

    arrays::matrix<dcomplex> density(gf_const_view<legendre> g);
    dcomplex density(gf_const_view<legendre, scalar_valued> g);

    //-------------------------------------------------------
    // For Real Frequency functions
    // ------------------------------------------------------

    arrays::matrix<dcomplex> density(gf_const_view<refreq> g, double beta);
    dcomplex density(gf_const_view<refreq, scalar_valued> g, double beta);

    arrays::matrix<dcomplex> density(gf_const_view<refreq> g);
    dcomplex density(gf_const_view<refreq, scalar_valued> g);

  } // namespace gfs

  namespace clef {
    TRIQS_CLEF_MAKE_FNT_LAZY(density);
  }
} // namespace triqs
