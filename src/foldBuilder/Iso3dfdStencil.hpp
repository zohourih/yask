/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2016, Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

*****************************************************************************/

// Implement isotropic 3D finite-difference stencil.
// 2nd-order in time.

#include "StencilBase.hpp"

class Iso3dfdStencil : public StencilBase {

    const int maxOrder = 16;
    
protected:
    double* _coeff;             // stencil coefficients.
    double _dxyz;               // dx, dy, dz factor.
    const Grid3d& _vel;           // const velocity grid.
    
public:

    Iso3dfdStencil(const Grid3d& vel, int order=8, double dxyz=50.0) :
        StencilBase(order), _coeff(0), _dxyz(dxyz), _vel(vel)
    {
        setOrder(order);
    }

    virtual ~Iso3dfdStencil() {
        if (_coeff)
            delete[] _coeff;
    }

    // Set order.
    // Return true if successful.
    virtual bool setOrder(int order) {
        _order = order;

        // support only certain orders.
        // Only order 8 and 16 use accurate coefficients; others are for debug.
        if (order % 2 == 1 || order > maxOrder)
            return false;

        // set the coefficients.
        if (_coeff)
            delete[] _coeff;
        int n = order / 2 + 1;
        _coeff = new double[n];

        double coeff8[] = {
            -2.847222222,
            +1.6,
            -0.2,
            +2.53968e-2,
            -1.785714e-3};

        double coeff16[] = {
            -3.0548446,
            +1.7777778,
            -3.1111111e-1,
            +7.572087e-2,
            -1.76767677e-2,
            +3.480962e-3,
            -5.180005e-4,
            +5.074287e-5,
            -2.42812e-6};

        double *coeffN = (order == 8) ? coeff8 : coeff16;

        // copy the coefficients and adjust using dx, dy, dz.
        double dxyz2 = _dxyz * _dxyz;
        for (int i = 0; i < n; i++)
            _coeff[i] = coeffN[i] / dxyz2;
        _coeff[0] *= 3.0;

        return true;
    }

    // Calculate and return the value of stencil at u(t2, v0, i, j, k)
    // based on u(t0, v0, ...);
    virtual GridValue value(Grid5d& u, int tW, int t0, int v0, int i, int j, int k) const {

        // if the wanted time (tW) is <= the last known time (t0),
        // we are done--just return the known value from the grid.
        if (tW <= t0)
            return u(tW, v0, i, j, k);

        // not known; calc using values at tW-1 and tW-2 recursively.
        int tm1 = tW - 1;  // one timestep ago.
        int tm2 = tW - 2;  // two timesteps ago.

        // start with center value.
        GridValue v = _coeff[0] * value(u, tm1, t0, v0, i, j, k);

        // add values from x, y, and z axes.
        for (int r = 1; r <= _order/2; r++) {
            
            v += (

                  // x-axis.
                  value(u, tm1, t0, v0, i-r, j, k) +
                  value(u, tm1, t0, v0, i+r, j, k) +
            
                  // y-axis.
                  value(u, tm1, t0, v0, i, j-r, k) +
                  value(u, tm1, t0, v0, i, j+r, k) +
            
                  // z-axis.
                  value(u, tm1, t0, v0, i, j, k-r) +
                  value(u, tm1, t0, v0, i, j, k+r)

                  ) * _coeff[r];
        }

        // temporal and velocity components.
        v = (2.0 * value(u, tm1, t0, v0, i, j, k)) 
            - value(u, tm2, t0, v0, i, j, k) // value from previous time.
            + (v * _vel(i, j, k));       // velocity.

        return v;
    }
};