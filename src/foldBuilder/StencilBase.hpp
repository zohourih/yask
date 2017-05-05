/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2017, Intel Corporation

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

// Base class for defining stencil equations.

#ifndef STENCIL_BASE
#define STENCIL_BASE

#include <map>
using namespace std;

namespace yask {

#define REGISTER_STENCIL(Class) static Class registered_ ## Class(stencils)

    typedef enum { STENCIL_CONTEXT } YASKSection;
    typedef vector<string> CodeList;
    typedef map<YASKSection, CodeList > ExtensionsList;

#define REGISTER_CODE_EXTENSION(section,code) _extensions[section].push_back(code);
#define REGISTER_STENCIL_CONTEXT_EXTENSION(...) REGISTER_CODE_EXTENSION(STENCIL_CONTEXT,#__VA_ARGS__)

    class StencilBase;
    typedef map<string, StencilBase*> StencilList;

    // An interface for all objects that participate in stencil definitions.
    // This allows a programmer to use object composition in addition to
    // inheritance to define stencils.
    class StencilPart {

    public:
        StencilPart() {}
        virtual ~StencilPart() {}

        // Return a reference to the main stencil object.
        virtual StencilBase& get_stencil_base() =0;
    };

    // The class all stencil problems must implement.
    class StencilBase : public StencilPart {
    protected:

        // Simple name for the stencil.
        string _name;
    
        // A grid is an n-dimensional tensor that is indexed by grid indices.
        // Vectorization will be applied to grid accesses.
        Grids _grids;       // keep track of all registered grids.

        // A parameter is an n-dimensional tensor that is NOT indexed by grid indices.
        // It is used to pass some sort of index-invarant setting to a stencil function.
        // Its indices must be resolved when define() is called.
        // At this time, this is not checked, so be careful!!
        Params _params;     // keep track of all registered non-grid vars.

        // All equations defined in this stencil.
        Eqs _eqs;
    
        // Code extensions that overload default functions from YASK in the generated code for this 
        // Stencil code
        ExtensionsList _extensions;

        // Initialize name.
        StencilBase(const string name) :
            _name(name) { }
    
    public:
        // Initialize name and register this new object in a list.
        StencilBase(const string name, StencilList& stencils) :
            _name(name)
        {
            stencils[_name] = this;
        }
        virtual ~StencilBase() { }

        // Return a reference to the main stencil object.
        virtual StencilBase& get_stencil_base() {
            return *this;
        }
    
        // Identification.
        virtual const string& getName() const { return _name; }
    
        // Get the registered grids and params.
        virtual Grids& getGrids() { return _grids; }
        virtual Grids& getParams() { return _params; }

        // Get the registered equations.
        virtual Eqs& getEqs() { return _eqs; }

        // Radius stub methods.
        virtual bool usesRadius() const { return false; }
        virtual bool setRadius(int radius) { return false; }
        virtual int getRadius() const { return 0; }

        // Define grid values relative to given offsets in each dimension.
        virtual void define(const IntTuple& offsets) = 0;

        // Get user-provided code for the given section.
        CodeList * getExtensionCode ( YASKSection section ) 
        { 
            auto elem = _extensions.find(section);
            if ( elem != _extensions.end() ) {
                return &elem->second;
            }
            return NULL;
        }
    };

    // A base class for stencils that have a 'radius'.
    class StencilRadiusBase : public StencilBase {
    protected:
        int _radius;         // stencil radius (for convenience; optional).

    public:
        StencilRadiusBase(const string name, StencilList& stencils, int radius) :
            StencilBase(name, stencils), _radius(radius) {}

        // Does use radius.
        virtual bool usesRadius() const { return true; }
    
        // Set radius.
        // Return true if successful.
        virtual bool setRadius(int radius) {
            _radius = radius;
            return radius >= 0;  // support only non-neg. radius.
        }

        // Get radius.
        virtual int getRadius() { return _radius; }
    };

    // A base class for stencils created via the YASK compiler API.
    class StencilSolution : public StencilBase,
                            public virtual stencil_solution {
    public:
        StencilSolution(const string& name) :
            StencilBase(name) { }
        virtual ~StencilSolution() {}

        virtual void set_name(std::string name) {
            _name = name;
        }
        virtual const std::string& get_name() const {
            return _name;
        }

        virtual grid_ptr new_grid(std::string name,
                                  std::string dim1 = "",
                                  std::string dim2 = "",
                                  std::string dim3 = "",
                                  std::string dim4 = "",
                                  std::string dim5 = "",
                                  std::string dim6 = "");

        virtual void add_equation(equation_node_ptr eq) {
            auto p = dynamic_pointer_cast<EqualsExpr>(eq);
            assert(p);
            _eqs.addEq(p);
        }
        virtual int get_num_equations() const {
            return _eqs.getNumEqs();
        }
        virtual equation_node_ptr get_equation(int n) {
            assert(n >= 0 && n < get_num_equations());
            return _eqs.getEqs().at(n);
        }

        // Equations normally created by 'define' must be
        // created via APIs.
        virtual void define(const IntTuple& offsets) {
            assert(0);
        }
    };
} // namespace yask.

#endif