/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2018, Intel Corporation

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

// Implement various example symmetric 3D stencil shapes that read and
// write from only one 3D grid.

#include "Soln.hpp"

class ExampleStencil : public StencilRadiusBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.
    MAKE_DOMAIN_INDEX(z);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x, y, z); // time-varying 3D grid.
    
    // Return a coefficient.  Note: This returns completely fabricated
    // values only for illustrative purposes; they have no mathematical
    // significance.
    virtual double coeff(int di, int dj, int dk) const {
        int sumAbs = abs(di) + abs(dj) + abs(dk);
        if (sumAbs == 0)
            return 0.9;
        double sumSq = double(di*di) + double(dj*dj) + double(dk*dk);
        double num = (sumAbs % 2 == 0) ? -0.8 : 0.8;
        return num / sumSq;
    }

    // Add additional points to expression v.
    virtual void addPoints(GridValue& v) =0;
    
public:
    ExampleStencil(const string& name, StencilList& stencils, int radius=2) :
        StencilRadiusBase(name, stencils, radius) { }

    // Define equation at t+1 based on values at t.
    virtual void define() {

        // start with center point.
        GridValue v = coeff(0, 0, 0) * data(t, x, y, z);

        // Add additional points from derived class.
        addPoints(v);

        // define the value at t+1 to be equivalent to v.
        data(t+1, x, y, z) EQUALS v;
    }
};

class ExampleStencil2D : public StencilRadiusBase {

protected:

    // Indices & dimensions.
    MAKE_STEP_INDEX(t);           // step in time dim.
    MAKE_DOMAIN_INDEX(x);         // spatial dim.
    MAKE_DOMAIN_INDEX(y);         // spatial dim.

    // Vars.
    MAKE_GRID(data, t, x, y);     // time-varying 2D grid.
    
    // Return a coefficient.  Note: This returns completely fabricated
    // values only for illustrative purposes; they have no mathematical
    // significance.
    virtual double coeff(int di, int dj) const {
        int sumAbs = abs(di) + abs(dj);
        if (sumAbs == 0)
            return 0.9;
        double sumSq = double(di*di) + double(dj*dj);
        double num = (sumAbs % 2 == 0) ? -0.8 : 0.8;
        return num / sumSq;
    }

    // Add additional points to expression v.
    virtual void addPoints(GridValue& v) =0;
    
public:
    ExampleStencil2D(const string& name, StencilList& stencils, int radius=2) :
        StencilRadiusBase(name, stencils, radius) { }

    // Define equation at t+1 based on values at t.
    virtual void define() {

        // start with center point.
        GridValue v = coeff(0, 0) * data(t, x, y);

        // Add additional points from derived class.
        addPoints(v);

        // define the value at t+1 to be equivalent to v.
        data(t+1, x, y) EQUALS v;
    }
};

// Add points from x, y, and z axes.
class AxisStencil : public ExampleStencil {
protected:

    // Add additional points to v.
    virtual void addPoints(GridValue& v)
    {
        for (int r = 1; r <= _radius; r++) {

            // On the axes, assume values are isotropic, i.e., the same
            // for all points the same distance from the origin.
            double c = coeff(r, 0, 0);
            v += c * 
                (
                 // x-axis.
                 data(t, x-r, y, z) +
                 data(t, x+r, y, z) +
                 
                 // y-axis.
                 data(t, x, y-r, z) +
                 data(t, x, y+r, z) +
                 
                 // z-axis.
                 data(t, x, y, z-r) +
                 data(t, x, y, z+r)
                 );
        }
    }

public:
    AxisStencil(StencilList& stencils, int radius=2) :
        ExampleStencil("3axis", stencils, radius) { }
    AxisStencil(const string& name, StencilList& stencils, int radius=2) :
        ExampleStencil(name, stencils, radius) { }
};

REGISTER_STENCIL(AxisStencil);

// Add points from x, y, and z axes.
class AxisModStencil : public ExampleStencil {
protected:

    // Add additional points to v.
    virtual void addPoints(GridValue& v)
    {
        for (int r = 1; r <= _radius; r++) {

            v += coeff(-r, 0, 0) * data(t, x-r, y, z);
            v += coeff(+r, 0, 0) * data(t, x+r, y, z);
            v += coeff(0, -r, 0) * data(t, x, y-r, z);
            v += coeff(0, +r, 0) * data(t, x, y+r, z);
            v += coeff(0, 0, -r) * data(t, x, y, z-r);
            v += coeff(0, 0, +r) * data(t, x, y, z+r);
        }
    }

public:
    AxisModStencil(StencilList& stencils, int radius=2) :
        ExampleStencil("3axismod", stencils, radius) { }
    AxisModStencil(const string& name, StencilList& stencils, int radius=2) :
        ExampleStencil(name, stencils, radius) { }
};

REGISTER_STENCIL(AxisModStencil);

// Add points from x and y axes.
class AxisModStencil2D : public ExampleStencil2D {
protected:

    // Add additional points to v.
    virtual void addPoints(GridValue& v)
    {
        for (int r = 1; r <= _radius; r++) {

            v += coeff(-r, 0) * data(t, x-r, y);
            v += coeff(+r, 0) * data(t, x+r, y);
            v += coeff(0, -r) * data(t, x, y-r);
            v += coeff(0, +r) * data(t, x, y+r);
        }
    }

public:
    AxisModStencil2D(StencilList& stencils, int radius=2) :
        ExampleStencil2D("2axismod", stencils, radius) { }
    AxisModStencil2D(const string& name, StencilList& stencils, int radius=2) :
        ExampleStencil2D(name, stencils, radius) { }
};

REGISTER_STENCIL(AxisModStencil2D);

// Add points from x-y, x-z, and y-z diagonals.
class DiagStencil : public AxisStencil {
protected:

    // Add additional points to v.
    virtual void addPoints(GridValue& v)
    {
        // Get points from axes.
        AxisStencil::addPoints(v);

        // Add points from diagonals.
        for (int r = 1; r <= _radius; r++) {

            // x-y diagonal.
            v += coeff(-r, -r, 0) * data(t, x-r, y-r, z);
            v += coeff(+r, -r, 0) * data(t, x+r, y-r, z);
            v -= coeff(-r, +r, 0) * data(t, x-r, y+r, z);
            v -= coeff(+r, +r, 0) * data(t, x+r, y+r, z);

            // x-z diagonal.
            v += coeff(-r, 0, -r) * data(t, x-r, y, z-r);
            v += coeff(+r, 0, +r) * data(t, x+r, y, z+r);
            v -= coeff(-r, 0, +r) * data(t, x-r, y, z+r);
            v -= coeff(+r, 0, -r) * data(t, x+r, y, z-r);

            // y-z diagonal.
            v += coeff(0, -r, -r) * data(t, x, y-r, z-r);
            v += coeff(0, +r, +r) * data(t, x, y+r, z+r);
            v -= coeff(0, -r, +r) * data(t, x, y-r, z+r);
            v -= coeff(0, +r, -r) * data(t, x, y+r, z-r);
        }
    }

public:
    DiagStencil(StencilList& stencils, int radius=2) :
        AxisStencil("9axis", stencils, radius) { }
    DiagStencil(const string& name, StencilList& stencils, int radius=2) :
        AxisStencil(name, stencils, radius) { }
};

REGISTER_STENCIL(DiagStencil);

// Add points from x-y, x-z, and y-z planes not covered by axes or diagonals.
class PlaneStencil : public DiagStencil {
protected:
    
    // Add additional points to v.
    virtual void addPoints(GridValue& v)
    {
        // Get points from axes and diagonals.
        DiagStencil::addPoints(v);

        // Add remaining points on planes.
        for (int r = 1; r <= _radius; r++) {
            for (int m = r+1; m <= _radius; m++) {

                // x-y plane.
                v += coeff(-r, -m, 0) * data(t, x-r, y-m, z);
                v += coeff(-m, -r, 0) * data(t, x-m, y-r, z);
                v += coeff(+r, +m, 0) * data(t, x+r, y+m, z);
                v += coeff(+m, +r, 0) * data(t, x+m, y+r, z);
                v -= coeff(-r, +m, 0) * data(t, x-r, y+m, z);
                v -= coeff(-m, +r, 0) * data(t, x-m, y+r, z);
                v -= coeff(+r, -m, 0) * data(t, x+r, y-m, z);
                v -= coeff(+m, -r, 0) * data(t, x+m, y-r, z);

                // x-z plane.
                v += coeff(-r, 0, -m) * data(t, x-r, y, z-m);
                v += coeff(-m, 0, -r) * data(t, x-m, y, z-r);
                v += coeff(+r, 0, +m) * data(t, x+r, y, z+m);
                v += coeff(+m, 0, +r) * data(t, x+m, y, z+r);
                v -= coeff(-r, 0, +m) * data(t, x-r, y, z+m);
                v -= coeff(-m, 0, +r) * data(t, x-m, y, z+r);
                v -= coeff(+r, 0, -m) * data(t, x+r, y, z-m);
                v -= coeff(+m, 0, -r) * data(t, x+m, y, z-r);

                // y-z plane.
                v += coeff(0, -r, -m) * data(t, x, y-r, z-m);
                v += coeff(0, -m, -r) * data(t, x, y-m, z-r);
                v += coeff(0, +r, +m) * data(t, x, y+r, z+m);
                v += coeff(0, +m, +r) * data(t, x, y+m, z+r);
                v -= coeff(0, -r, +m) * data(t, x, y-r, z+m);
                v -= coeff(0, -m, +r) * data(t, x, y-m, z+r);
                v -= coeff(0, +r, -m) * data(t, x, y+r, z-m);
                v -= coeff(0, +m, -r) * data(t, x, y+m, z-r);
            }
        }
    }

public:
    PlaneStencil(StencilList& stencils, int radius=2) :
        DiagStencil("3plane", stencils, radius) { }
    PlaneStencil(const string& name, StencilList& stencils, int radius=2) :
        DiagStencil(name, stencils, radius) { }
};

REGISTER_STENCIL(PlaneStencil);

// Add points from rest of cube.
class CubeStencil : public PlaneStencil {
protected:

    // Add additional points to v.
    virtual void addPoints(GridValue& v)
    {
        // Get points from planes.
        PlaneStencil::addPoints(v);

        // Add points from rest of cube.
        for (int rx = 1; rx <= _radius; rx++)
            for (int ry = 1; ry <= _radius; ry++)
                for (int rz = 1; rz <= _radius; rz++) {

                    // Each quadrant.
                    v += coeff(rx, ry, rz) * data(t, x+rx, y+ry, z+rz);
                    v += coeff(rx, -ry, -rz) * data(t, x+rx, y-ry, z-rz);
                    v -= coeff(rx, ry, -rz) * data(t, x+rx, y+ry, z-rz);
                    v -= coeff(rx, -ry, rz) * data(t, x+rx, y-ry, z+rz);
                    v += coeff(-rx, ry, rz) * data(t, x-rx, y+ry, z+rz);
                    v += coeff(-rx, -ry, -rz) * data(t, x-rx, y-ry, z-rz);
                    v -= coeff(-rx, ry, -rz) * data(t, x-rx, y+ry, z-rz);
                    v -= coeff(-rx, -ry, rz) * data(t, x-rx, y-ry, z+rz);
                }
    }

public:
    CubeStencil(StencilList& stencils, int radius=1) :
        PlaneStencil("cube", stencils, radius) { }
    CubeStencil(const string& name, StencilList& stencils, int radius=1) :
        PlaneStencil(name, stencils, radius) { }
};

REGISTER_STENCIL(CubeStencil);
