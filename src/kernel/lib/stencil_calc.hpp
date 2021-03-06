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

#pragma once

namespace yask {
    
    /// Classes that support evaluation of one stencil group.
    /// A context contains of one or more groups.

    // Types of dependencies.
    enum DepType {
        certain_dep,
        possible_dep,
        num_deps
    };

    // A pure-virtual class base for a stencil group.
    class StencilGroupBase : public BoundingBox {
    protected:
        StencilContext* _generic_context = 0;
        std::string _name;
        int _scalar_fp_ops = 0;
        int _scalar_points_read = 0;
        int _scalar_points_written = 0;

        // Position of inner dim in stencil-dims tuple.
        int _inner_posn = 0;

        // Groups that this one depends on.
        std::map<DepType, StencilGroupSet> _depends_on;

        // Normalize the indices, i.e., subtract the rank offset
        // and divide by vector len in each dim.
        void normalize_indices(const Indices& orig, Indices& norm) const;
        
    public:

        // Grids that are written to by these stencils.
        GridPtrs outputGridPtrs;

        // Grids that are read by these stencils (not necessarify
        // read-only, i.e., a grid can be input and output).
        GridPtrs inputGridPtrs;

        // ctor, dtor.
        StencilGroupBase(StencilContext* context) :
            _generic_context(context) {

            // Make sure map entries exist.
            for (DepType dt = certain_dep; dt < num_deps; dt = DepType(dt+1)) {
                _depends_on[dt];
            }

            // Find index posn of inner loop.
            auto dims = context->get_dims();
            int ndims = dims->_stencil_dims.getNumDims();
            for (int i = 0; i < ndims; i++) {
                auto& dname = dims->_stencil_dims.getDimName(i);
                if (dname == dims->_inner_dim) {
                    _inner_posn = i;
                    break;
                }
            }
        }

        virtual ~StencilGroupBase() { }

        // Access to dims and MPI info.
        virtual DimsPtr get_dims() {
            return _generic_context->get_dims();
        }
        virtual MPIInfoPtr get_mpi_info() {
            return _generic_context->get_mpi_info();
        }

        // Get name of this group.
        virtual const std::string& get_name() { return _name; }

        // Get estimated number of FP ops done for one scalar eval.
        virtual int get_scalar_fp_ops() { return _scalar_fp_ops; }

        // Get number of points read and written for one scalar eval.
        virtual int get_scalar_points_read() const { return _scalar_points_read; }
        virtual int get_scalar_points_written() const { return _scalar_points_written; }

        // Add dependency.
        virtual void add_dep(DepType dt, StencilGroupBase* eg) {
            _depends_on.at(dt).insert(eg);
        }

        // Get dependencies.
        virtual const StencilGroupSet& get_deps(DepType dt) const {
            return _depends_on.at(dt);
        }
    
        // Set the bounding-box vars for this group in this rank.
        virtual void find_bounding_box();

        // Determine whether indices are in [sub-]domain.
        virtual bool
        is_in_valid_domain(const Indices& idxs) =0;

        // Calculate one scalar result at time t.
        virtual void
        calc_scalar(const Indices& idxs) =0;

        // Calculate results within a block.
        // Each block is typically computed in a separate OpenMP thread team.
        virtual void
        calc_block(const ScanIndices& region_idxs);

        // Calculate results within a sub-block.
        // Each sub-block is typically computed in a separate nested OpenMP thread.
        virtual void
        calc_sub_block(const ScanIndices& block_idxs);

        // Calculate a series of cluster results within an inner loop.
        // All indices start at 'start_idxs'. Inner loop iterates to
        // 'stop_inner' by 'step_inner'.
        // Indices must be rank-relative.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void
        calc_loop_of_clusters(const Indices& start_idxs, idx_t stop_inner) =0;

        // Calculate a series of cluster results within an inner loop.
        // The 'loop_idxs' must specify a range only in the inner dim.
        // Indices must be rank-relative.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        virtual void
        calc_loop_of_clusters(const ScanIndices& loop_idxs);

        // Calculate a series of vector results within an inner loop.
        // All indices start at 'start_idxs'. Inner loop iterates to
        // 'stop_inner' by 'step_inner'.
        // Indices must be rank-relative.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        // Each vector write is masked by 'write_mask'.
        virtual void
        calc_loop_of_vectors(const Indices& start_idxs, idx_t stop_inner,
                             idx_t write_mask) =0;

        // Calculate a series of vector results within an inner loop.
        // The 'loop_idxs' must specify a range only in the inner dim.
        // Indices must be rank-relative.
        // Indices must be normalized, i.e., already divided by VLEN_*.
        // Each vector write is masked by 'write_mask'.
        virtual void
        calc_loop_of_vectors(const ScanIndices& loop_idxs,
                             idx_t write_mask);
    };

} // yask namespace.
