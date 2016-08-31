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

// Stencil types.
#include "stencil.hpp"

// Base classes for stencil code.
#include "stencil_calc.hpp"

#include <sstream>
using namespace std;

namespace yask {

    // Initialize stencil equations and context.
    void StencilEquations::init(StencilContext& context)
    {
        if (_isInit)
            return;

        // Init each stencil equation.
        for (auto stencil : stencils)
            stencil->init(context);

        // Determine spatial skewing angles based on the
        // halos.  This assumes the smallest granularity of calculation is
        // CPTS_* in each dim.
        // These angles are used for both regions and blocks.
        context.angle_n = ROUND_UP(context.hn, CPTS_N);
        context.angle_x = ROUND_UP(context.hx, CPTS_X);
        context.angle_y = ROUND_UP(context.hy, CPTS_Y);
        context.angle_z = ROUND_UP(context.hz, CPTS_Z);
        cout << "Temporal skewing angles: " <<
            context.angle_n << ", " << context.angle_x << ", " << context.angle_y << ", " << context.angle_z << endl;

        // Temporal blocking only supported in x, y, z dims.
        if (context.rt > 1 && context.bt > 1 && context.angle_n > 0) {
            cerr << "Sorry, temporal cache-blocking is not yet supported in the 'n' dimension." << endl;
            exit(1);
        }

        // Determine max temporal-block time size.
        // This is the max "height" of the hyper-pyramid
        // based on its spatial base sizes and angles.
        idx_t max_bt = min(context.bt, context.rt);
        if (context.angle_n > 0)
            max_bt = min(max_bt, context.bn / context.angle_n / 2 + 1);
        if (context.angle_x > 0)
            max_bt = min(max_bt, context.bx / context.angle_x / 2 + 1);
        if (context.angle_y > 0)
            max_bt = min(max_bt, context.by / context.angle_y / 2 + 1);
        if (context.angle_z > 0)
            max_bt = min(max_bt, context.bz / context.angle_z / 2 + 1);
        cout << "Maximum allowed cache-block time-steps (bt): " << max_bt << endl;
        if (max_bt < context.bt) 
            context.bt = max_bt;
            cout << "Actual cache-block time-steps (bt): " << context.bt << endl;

        // We only need non-zero *region* angles if the region size is less than the rank size,
        // i.e., if the region covers the whole rank in a given dimension, no wave-front
        // is needed in thar dim.
        // These angles are used for regions boundaries only. In a given dimension,
        // a block may have non-zero angles, but its region may not.
        // TODO: make this grid-specific.
        context.rangle_n = (context.rn < context.dn) ? context.angle_n : 0;
        context.rangle_x = (context.rx < context.dx) ? context.angle_x : 0;
        context.rangle_y = (context.ry < context.dy) ? context.angle_y : 0;
        context.rangle_z = (context.rz < context.dz) ? context.angle_z : 0;
        cout << "Wave-front region angles: :" <<
            context.rangle_n << ", " << context.rangle_x << ", " << context.rangle_y << ", " << context.rangle_z << endl;

        _isInit = true;
    }


    ///// Top-level methods for evaluating reference and optimized stencils.

    // Eval stencil(s) over grid(s) using scalar code.
    void StencilEquations::calc_rank_ref(StencilContext& context)
    {
        init(context);
    
        // Start at a positive point to avoid any calculation referring
        // to negative time.
        idx_t t0 = TIME_DIM_SIZE * 2;

        TRACE_MSG("calc_problem_ref(%ld..%ld, 0..%ld, 0..%ld, 0..%ld, 0..%ld)", 
                  t0, t0 + context.dt - 1,
                  context.dn - 1,
                  context.dx - 1,
                  context.dy - 1,
                  context.dz - 1);
    
        // Time steps.
        // TODO: check that scalar version actually does CPTS_T time steps.
        // (At this point, CPTS_T == 1 for all existing stencil examples.)
        for(idx_t t = t0; t < t0 + context.dt; t += CPTS_T) {

            // equations to evaluate (only one in most stencils).
            for (auto stencil : stencils) {

                // Halo exchange for grid(s) updated by this equation.
                stencil->exchange_halos(context, t, t + CPTS_T);
            
                // grid index (only one in most stencils).
                for (idx_t n = 0; n < context.dn; n++) {

#pragma omp parallel for
                    for(idx_t ix = 0; ix < context.dx; ix++) {

                        CREW_FOR_LOOP
                            for(idx_t iy = 0; iy < context.dy; iy++) {

                                for(idx_t iz = 0; iz < context.dz; iz++) {

                                    TRACE_MSG("%s.calc_scalar(%ld, %ld, %ld, %ld, %ld)", 
                                              stencil->get_name().c_str(), t, n, ix, iy, iz);
                            
                                    // Evaluate the reference scalar code.
                                    stencil->calc_scalar(context, t, n, ix, iy, iz);
                                }
                            }
                    }
                }
            }
        } // iterations.
    }


    // Eval stencil(s) over grid(s) using optimized code.
    void StencilEquations::calc_rank_opt(StencilContext& context)
    {
        init(context);

        // Problem begin points.
        // Start at a positive time point to avoid any calculation referring
        // to negative time.
        idx_t begin_dt = TIME_DIM_SIZE * 2;
        idx_t begin_dn = 0, begin_dx = 0, begin_dy = 0, begin_dz = 0;
    
        // Problem end-points.
        idx_t end_dt = begin_dt + context.dt;
        idx_t end_dn = context.dn;
        idx_t end_dx = context.dx;
        idx_t end_dy = context.dy;
        idx_t end_dz = context.dz;

        TRACE_MSG("calc_rank_opt(%ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld)", 
                  begin_dt, end_dt-1,
                  begin_dn, end_dn-1,
                  begin_dx, end_dx-1,
                  begin_dy, end_dy-1,
                  begin_dz, end_dz-1);
    
        // Steps are based on region sizes.
        idx_t step_dt = context.rt;
        idx_t step_dn = context.rn;
        idx_t step_dx = context.rx;
        idx_t step_dy = context.ry;
        idx_t step_dz = context.rz;

        // Extend end points for regions due to wavefront angle.
        // For each subsequent time step in a region, the spatial location of
        // each block evaluation is shifted by the angle for each stencil. So,
        // the total shift in a region is the angle * num stencils * num
        // timesteps. Thus, the number of overlapping regions is ceil(total
        // shift / region size).  This assumes stencils are inter-dependent.
        // TODO: calculate stencil inter-dependency in the foldBuilder for each
        // dimension.
        idx_t nshifts = (idx_t(stencils.size()) * context.rt) - 1;
        end_dn += context.rangle_n * nshifts;
        end_dx += context.rangle_x * nshifts;
        end_dy += context.rangle_y * nshifts;
        end_dz += context.rangle_z * nshifts;
        TRACE_MSG("extended domain after wave-front adjustment for %ld shift(s): %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld", 
            nshifts,
                  begin_dt, end_dt-1,
                  begin_dn, end_dn-1,
                  begin_dx, end_dx-1,
                  begin_dy, end_dy-1,
                  begin_dz, end_dz-1);

        // Number of iterations to get from begin_dt to (but not including) end_dt,
        // stepping by step_dt.
        const idx_t num_dt = ((end_dt - begin_dt) + (step_dt - 1)) / step_dt;
        for (idx_t index_dt = 0; index_dt < num_dt; index_dt++)
        {
            // This value of index_dt covers dt from start_dt to stop_dt-1.
            const idx_t start_dt = begin_dt + (index_dt * step_dt);
            const idx_t stop_dt = min(start_dt + step_dt, end_dt);

            // If doing only one time step in a region (default), loop through equations here,
            // and do only one equation at a time in calc_region().
            if (step_dt == 1) {

                for (auto stencil : stencils) {

                    // Halo exchange for grid(s) updated by this equation.
                    stencil->exchange_halos(context, start_dt, stop_dt);

                    // Eval this stencil in calc_region().
                    StencilSet stencil_set;
                    stencil_set.insert(stencil);

                    // Include automatically-generated loop code that calls calc_region() for each region.
#include "stencil_rank_loops.hpp"
                }
            }

            // If doing more than one time step in a region,
            // must do all equations in calc_region().
            // TODO: allow doing all equations in region even with one time step for testing.
            else {

                StencilSet stencil_set;
                for (auto stencil : stencils) {

                    // Halo exchange for grid(s) updated by this equation.
                    stencil->exchange_halos(context, start_dt, stop_dt);
                    
                    // Make set of all equations.
                    stencil_set.insert(stencil);
                }
            
                // Include automatically-generated loop code that calls calc_region() for each region.
#include "stencil_rank_loops.hpp"
            }

        }
    }

    // Calculate results within a region.
    // Each region is typically computed in a separate OpenMP 'for' region.
    // In it, we loop over the time steps and the stencil
    // equations and evaluate the blocks in the region.
    void StencilEquations::
    calc_region(StencilContext& context,
        
        // temporal range for this block.
        idx_t begin_rt, idx_t end_rt,

        // stencils to eval.
        StencilSet& stencil_set,

        // initial boundaries of region (for 1st stencil at begin_rt). 
        idx_t begin_rn0, idx_t begin_rx0, idx_t begin_ry0, idx_t begin_rz0,
        idx_t end_rn0, idx_t end_rx0, idx_t end_ry0, idx_t end_rz0)
    {
        TRACE_MSG("calc_region(%ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld)",
            begin_rt, end_rt - 1,
            begin_rn0, end_rn0 - 1,
            begin_rx0, end_rx0 - 1,
            begin_ry0, end_ry0 - 1,
            begin_rz0, end_rz0 - 1);

        // Set number of threads for a region.
        context.set_region_threads();

        // Set begin and end of region used by stencil_region_loops.hpp.
        // These will be adjusted by the temporal shift angles if applicable.
        idx_t begin_rn = begin_rn0;
        idx_t end_rn = end_rn0;
        idx_t begin_rx = begin_rx0;
        idx_t end_rx = end_rx0;
        idx_t begin_ry = begin_ry0;
        idx_t end_ry = end_ry0;
        idx_t begin_rz = begin_rz0;
        idx_t end_rz = end_rz0;

        // Steps within a region are based on block sizes.
        const idx_t step_rt = context.bt;
        const idx_t step_rn = context.bn;
        const idx_t step_rx = context.bx;
        const idx_t step_ry = context.by;
        const idx_t step_rz = context.bz;

        // Temporal shifts per block.
        idx_t nshifts = idx_t(stencil_set.size()) * context.bt;

        // Number of iterations to get from begin_rt to (but not including) end_rt,
        // stepping by step_rt.
        const idx_t num_rt = ((end_rt - begin_rt) + (step_rt - 1)) / step_rt;

        // Step through time steps in this region.
        for (idx_t index_rt = 0; index_rt < num_rt; index_rt++) {

            // This value of index_rt covers rt from start_rt to stop_rt-1.
            const idx_t start_rt = begin_rt + (index_rt * step_rt);
            const idx_t stop_rt = min(start_rt + step_rt, end_rt);

            // There are 4 phases required to properly tesselate 4D space
            // when using 3D temporal blocks (ignoring 'n' because it must have 0 angle).
            idx_t nphases = (context.bt > 1) ? 4 : 1;
            for (idx_t phase = 0; phase < nphases; phase++) {

                // Include automatically-generated loop code that calls
                // calc_temporal_block() for each block in this region.  Loops
                // through rn from begin_rn to end_rn-1; similar for rx, ry,
                // and rz.  This code typically contains OpenMP loop(s)
                // such that blocks are evaluated in parallel.
#include "stencil_region_loops.hpp"
            }

            // Shift spatial region boundaries for next iteration to
            // implement temporal wavefront.  We only shift backward, so
            // region loops must increment. They may do so in any order.
            begin_rn -= context.rangle_n * nshifts;
            end_rn -= context.rangle_n * nshifts;
            begin_rx -= context.rangle_x * nshifts;
            end_rx -= context.rangle_x * nshifts;
            begin_ry -= context.rangle_y * nshifts;
            end_ry -= context.rangle_y * nshifts;
            begin_rz -= context.rangle_z * nshifts;
            end_rz -= context.rangle_z * nshifts;

        } // time.

        // Reset threads back to max.
        context.set_max_threads();
    }

    // Calculate results in one temporal block.
    // A temporal block is composed of one or more spatial blocks.
    // There will be exactly one spatial block if temporal blocking
    // is not being used, i.e., bt==1 and/or rt==1.
    // Typically, this is called in parallel for many blocks via OpenMP.
    // Then, within a spatial block, nested OpenMP may be used.
    void StencilEquations::calc_temporal_block(StencilContext& context,

        // Temporal range for this block.
        idx_t begin_bt, idx_t end_bt,

        // Tesselation phase.
        idx_t phase,

        // Stencils to eval.
        StencilSet& stencil_set,

        // Initial boundaries of region for this block (for 1st stencil at begin_bt).
        // If rt > bt, the regions boundaries at one time will NOT be the same
        // as those from a different time due to the region skewing angles.
        idx_t begin_rn0, idx_t begin_rx0, idx_t begin_ry0, idx_t begin_rz0,
        idx_t end_rn0, idx_t end_rx0, idx_t end_ry0, idx_t end_rz0,

        // Initial boundaries of block in the region (for 1st stencil at begin_bt).
        // If rt > bt, the blocks at one time will NOT "line up" with the blocks
        // from a different time.
        idx_t begin_bn0, idx_t begin_bx0, idx_t begin_by0, idx_t begin_bz0,
        idx_t end_bn0, idx_t end_bx0, idx_t end_by0, idx_t end_bz0)
    {
        TRACE_MSG("calc_temporal_block(phase %ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld, %ld..%ld)",
            phase,
            begin_bt, end_bt - 1,
            begin_bn0, end_bn0 - 1,
            begin_bx0, end_bx0 - 1,
            begin_by0, end_by0 - 1,
            begin_bz0, end_bz0 - 1);

        // Determine which (if any) edges of this block (begin_bn0..end_bz0) are against 
        // the region boundary (begin_rn0..end_rz0).
        // In other words, is this a starting and/or ending block in
        // the given direction(s)?
        bool at_begin_rn = begin_bn0 == begin_rn0;
        bool at_end_rn = end_bn0 == end_rn0;
        bool at_begin_rx = begin_bx0 == begin_rx0;
        bool at_end_rx = end_bx0 == end_rx0;
        bool at_begin_ry = begin_by0 == begin_ry0;
        bool at_end_ry = end_by0 == end_ry0;
        bool at_begin_rz = begin_bz0 == begin_rz0;
        bool at_end_rz = end_bz0 == end_rz0;

        // Steps within a block are based on the cluster size.
        const idx_t step_bt = CPTS_T; // usually 1.
        const idx_t step_bn = CPTS_N;
        const idx_t step_bx = CPTS_X;
        const idx_t step_by = CPTS_Y;
        const idx_t step_bz = CPTS_Z;

        // How many parts are in this phase?
        idx_t nparts = (phase == 1 || phase == 2) ? 3 : 1;

        // Loop through parts.
        for (idx_t part = 0; part < nparts; part++) {

            // Temporal shift counter.
            idx_t tshift = 0;

            // Number of iterations to get from begin_bt to (but not including) end_bt,
            // stepping by step_bt.
            const idx_t num_bt = ((end_bt - begin_bt) + (step_bt - 1)) / step_bt;

            // Step through time steps in this block.
            for (idx_t index_bt = 0; index_bt < num_bt; index_bt++) {

                // This value of index_bt covers bt from start_bt to stop_bt-1.
                const idx_t start_bt = begin_bt + (index_bt * step_bt);
                const idx_t stop_bt = min(start_bt + step_bt, end_bt);

                // Equations to evaluate at these time step(s).
                // Shifting occurs in this loop to support staggered grids.
                for (auto stencil : stencils) {
                    if (stencil_set.count(stencil)) {

                        // Calculate boundaries of *region* for this spatial block.
                        // The original parameters, begin_rn..end_rz describe the boundaries
                        // of the region at begin_bt only. In addition, they are may extend
                        // outside of the rank domain.
                        // Now, we need to shift begin_rn..end_rz due to any region angles
                        // rangle_n..rangle_z and clamp the boundaries to the rank domain.
                        // All these shifts are negative because these are region boundaries
                        // (see calc_region()).
                        // TODO: may need to enhance this when MPI enabled with temporal blocking.
                        idx_t begin_rn = max(begin_rn0 - context.rangle_n * tshift, idx_t(0));
                        idx_t end_rn = min(end_rn0 - context.rangle_n * tshift, context.dn);
                        idx_t begin_rx = max(begin_rx0 - context.rangle_x * tshift, idx_t(0));
                        idx_t end_rx = min(end_rx0 - context.rangle_x * tshift, context.dx);
                        idx_t begin_ry = max(begin_ry0 - context.rangle_y * tshift, idx_t(0));
                        idx_t end_ry = min(end_ry0 - context.rangle_y * tshift, context.dy);
                        idx_t begin_rz = max(begin_rz0 - context.rangle_z * tshift, idx_t(0));
                        idx_t end_rz = min(end_rz0 - context.rangle_z * tshift, context.dz);

                        // Shift amounts in this block for this time and stencil.
                        idx_t shift_n = context.angle_n * tshift;
                        idx_t shift_x = context.angle_x * tshift;
                        idx_t shift_y = context.angle_y * tshift;
                        idx_t shift_z = context.angle_z * tshift;

                        // The spatial block is described by begin_bn..end_rz.
                        // The phase and part determine what shape is created.
                        // Phase 0: start with spatial block that fills space; shrink
                        // it (increase begin and decrease end) each time-step.
                        idx_t begin_bn = begin_bn0 + shift_n;
                        idx_t end_bn = end_bn0 - shift_n;
                        idx_t begin_bx = begin_bx0 + shift_x;
                        idx_t end_bx = end_bx0 - shift_x;
                        idx_t begin_by = begin_by0 + shift_y;
                        idx_t end_by = end_by0 - shift_y;
                        idx_t begin_bz = begin_bz0 + shift_z;
                        idx_t end_bz = end_bz0 - shift_z;

                        // If at any region boundary, reset block boundary to match it.
                        if (at_begin_rn) begin_bn = begin_rn;
                        if (at_end_rn) end_bn = end_rn;
                        if (at_begin_rx) begin_bx = begin_rx;
                        if (at_end_rx) end_bx = end_rx;
                        if (at_begin_ry) begin_by = begin_ry;
                        if (at_end_ry) end_by = end_ry;
                        if (at_begin_rz) begin_bz = begin_rz;
                        if (at_end_rz) end_bz = end_rz;

                        // Determine beginning of *next* phase-0 block in each dim.
                        // This will be needed by phases 1-3.
                        // The next block will be created by a different call to
                        // calc_temporal_block().
                        idx_t begin_next_bx = end_bx0 + shift_x;
                        idx_t begin_next_by = end_by0 + shift_y;
                        idx_t begin_next_bz = end_bz0 + shift_z;

                        // Phase 1: fill in between phase-0 shapes.
                        // Start with space between a pair of shapes and expand it.
                        if (phase == 1) {

                            // Part 0 (x): n, y, and z dims track phase 0, but
                            // x dim grows to fill space between phase-0 shape in this
                            // block and phase-0 shape in *next* block in x-dim.
                            if (part == 0) {

                                // Start at end of phase-0 x-face, calculated above.
                                begin_bx = end_bx; 

                                // Extend to next phase-0 x-face.
                                end_bx = begin_next_bx;
                            }

                            // Part 1 (y): n, x, and z dims track phase 0, but
                            // y dim grows to fill space between phase-0 shape in this
                            // block and phase-0 shape in *next* block in y-dim.
                            else if (part == 1) {

                                // Start at end of phase-0 y-face, calculated above.
                                begin_by = end_by;

                                // Extend to next phase-0 y-face.
                                end_by = begin_next_by;
                            }

                            // Part 2 (z): n, x, and y dims track phase 0, but
                            // z dim grows to fill space between phase-0 shape in this
                            // block and phase-0 shape in *next* block in z-dim.
                            else if (part == 2) {

                                // Start at end of phase-0 z-face, calculated above.
                                begin_bz = end_bz;

                                // Extend to next phase-0 z-face.
                                end_bz = begin_next_bz;
                            }
                        }

                        // Phase 2: fill in between phase-1 shapes.
                        else if (phase == 2) {

                            // Part 0 (xy): n and z dims track phase 1, but
                            // x and y dims grow to fill space between x and y
                            // phase-1 shapes in this block and next x and y
                            // phase-1 shapes in *next* block.
                            if (part == 0) {

                                // Start and end of phase-0 x and y faces, calculated above.
                                begin_bx = end_bx;
                                begin_by = end_by;

                                // Extend to phase-0 x and y faces of next block.
                                end_bx = begin_next_bx;
                                end_by = begin_next_by;
                            }

                            // Part 1 (xz): n and y dims track phase 1, but
                            // x and z dims grow to fill space between x and z
                            // phase-1 shapes in this block and next x and z
                            // phase-1 shapes in *next* block.
                            else if (part == 1) {

                                // Start and end of phase-0 x and z faces, calculated above.
                                begin_bx = end_bx;
                                begin_bz = end_bz;

                                // Extend to phase-0 x and z faces of next block.
                                end_bx = begin_next_bx;
                                end_bz = begin_next_bz;
                            }

                            // Part 2 (yz): n and x dims track phase 1, but
                            // y and z dims grow to fill space between y and z
                            // phase-1 shapes in this block and next y and z
                            // phase-1 shapes in *next* block.
                            else if (part == 2) {

                                // Start and end of phase-0 y and z faces, calculated above.
                                begin_by = end_by;
                                begin_bz = end_bz;

                                // Extend to phase-0 y and z faces of next block.
                                end_by = begin_next_by;
                                end_bz = begin_next_bz;
                            }
                        }

                        // Phase 3: fill in between phase-2 shapes.
                        // This will grow in all dimensions (opposite of phase-0).
                        // TODO: fuse phase 3 with phase 1 to create hyper-diamonds
                        // instead of hyper-pyramids. Will require alignment between
                        // blocks.
                        else if (phase == 3) {

                            // Start and end of phase-0 faces, calculated above.
                            begin_bx = end_bx;
                            begin_by = end_by;
                            begin_bz = end_bz;

                            // Extend to phase-0 faces of next block.
                            end_bx = begin_next_bx;
                            end_by = begin_next_by;
                            end_bz = begin_next_bz;
                        }

                        // Always clamp to region boundaries.
                        begin_bn = max(begin_bn, begin_rn);
                        end_bn = min(end_bn, end_rn);
                        begin_bx = max(begin_bx, begin_rx);
                        end_bx = min(end_bx, end_rx);
                        begin_by = max(begin_by, begin_ry);
                        end_by = min(end_by, end_ry);
                        begin_bz = max(begin_bz, begin_rz);
                        end_bz = min(end_bz, end_rz);

                        // Only need to loop through the block if all ranges are > 0.
                        if (end_bn > begin_bn &&
                            end_bx > begin_bx &&
                            end_by > begin_by &&
                            end_bz > begin_bz) {

                            // Calculate a spatial block at time start_bt.
                            stencil->calc_spatial_block(context, start_bt,
                                begin_bn, begin_bx, begin_by, begin_bz,
                                end_bn, end_bx, end_by, end_bz);
                        }

                        tshift++;
                    } // stencil in set.
                } // stencil equations.
            } // time.
        } // part of phase.
    }


    // Exchange halo data for the given time.
    void StencilBase::exchange_halos(StencilContext& context, idx_t begin_rt, idx_t end_rt)
    {
#ifdef USE_MPI
        TRACE_MSG("exchange_halos(%ld..%ld)", begin_rt, end_rt);
        double start_time = getTimeInSecs();

        // These vars control blocking within halo packing.
        // Currently, only zv has a loop in the calc_halo macros below.
        // Thus, step_{n,x,y}v must be 1.
        // TODO: make step_zv a parameter.
        const idx_t step_nv = 1;
        const idx_t step_xv = 1;
        const idx_t step_yv = 1;
        const idx_t step_zv = 4;

        // List of grids updated by this equation.
        // These are the grids that need their halos exchanged.
        auto eqGridPtrs = getEqGridPtrs();

        // TODO: put this loop inside visitNeighbors.
        for (size_t gi = 0; gi < eqGridPtrs.size(); gi++) {

            // Get pointer to generic grid and derived type.
            // TODO: Make this more general.
            auto gp = eqGridPtrs[gi];
#if USING_DIM_N
            auto gpd = dynamic_cast<Grid_TNXYZ*>(gp);
#else
            auto gpd = dynamic_cast<Grid_TXYZ*>(gp);
#endif
            assert(gpd);

            // Determine halo sizes to be exchanged for this grid;
            // context.h* contains the max value across all grids.  The grid
            // contains the halo+pad size actually allocated.
            // Since neither of these is exactly what we want, we use
            // the minimum of these values as a conservative value. TODO:
            // Store the actual halo needed in each grid and use this.
#if USING_DIM_N
            idx_t hn = min(context.hn, gpd->get_pn());
#else
            idx_t hn = 0;
#endif
            idx_t hx = min(context.hx, gpd->get_px());
            idx_t hy = min(context.hy, gpd->get_py());
            idx_t hz = min(context.hz, gpd->get_pz());
            
            // Array to store max number of request handles.
            MPI_Request reqs[StencilContext::Bufs::nBufDirs * context.neighborhood_size];
            int nreqs = 0;

            // Pack data and initiate non-blocking send/receive to/from all neighbors.
            TRACE_MSG("rank %i: exchange_halos: packing data for grid '%s'...",
                      context.my_rank, gp->get_name().c_str());
            context.bufs[gp].visitNeighbors
                (context,
                 [&](idx_t nn, idx_t nx, idx_t ny, idx_t nz,
                     int neighbor_rank,
                     Grid_NXYZ* sendBuf,
                     Grid_NXYZ* rcvBuf)
                 {
                     // Pack and send data if buffer exists.
                     if (sendBuf) {

                         // Set begin/end vars to indicate what part
                         // of main grid to read from.
                         // Init range to whole rank size (inside halos).
                         idx_t begin_n = 0;
                         idx_t begin_x = 0;
                         idx_t begin_y = 0;
                         idx_t begin_z = 0;
                         idx_t end_n = context.dn;
                         idx_t end_x = context.dx;
                         idx_t end_y = context.dy;
                         idx_t end_z = context.dz;

                         // Modify begin and/or end based on direction.
                         if (nn == idx_t(context.rank_prev)) // neighbor is prev N.
                             end_n = hn; // read first halo-width only.
                         if (nn == idx_t(context.rank_next)) // neighbor is next N.
                             begin_n = context.dn - hn; // read last halo-width only.
                         if (nx == idx_t(context.rank_prev)) // neighbor is on left.
                             end_x = hx;
                         if (nx == idx_t(context.rank_next)) // neighbor is on right.
                             begin_x = context.dx - hx;
                         if (ny == idx_t(context.rank_prev)) // neighbor is in front.
                             end_y = hy;
                         if (ny == idx_t(context.rank_next)) // neighbor is in back.
                             begin_y = context.dy - hy;
                         if (nz == idx_t(context.rank_prev)) // neighbor is above.
                             end_z = hz;
                         if (nz == idx_t(context.rank_next)) // neighbor is below.
                             begin_z = context.dz - hz;

                         // Divide indices by vector lengths.
                         // Begin/end vars shouldn't be negative, so '/' is ok.
                         idx_t begin_nv = begin_n / VLEN_N;
                         idx_t begin_xv = begin_x / VLEN_X;
                         idx_t begin_yv = begin_y / VLEN_Y;
                         idx_t begin_zv = begin_z / VLEN_Z;
                         idx_t end_nv = end_n / VLEN_N;
                         idx_t end_xv = end_x / VLEN_X;
                         idx_t end_yv = end_y / VLEN_Y;
                         idx_t end_zv = end_z / VLEN_Z;

                         // TODO: fix this when MPI + wave-front is enabled.
                         idx_t t = begin_rt;
                         
                         // Define calc_halo() to copy a vector from main grid to sendBuf.
                         // Index sendBuf using index_* vars because they are zero-based.
#define calc_halo(context, t,                                           \
                  start_nv, start_xv, start_yv, start_zv,               \
                  stop_nv, stop_xv, stop_yv, stop_zv)  do {             \
                         idx_t nv = start_nv;                           \
                         idx_t xv = start_xv;                           \
                         idx_t yv = start_yv;                           \
                         idx_t izv = index_zv * step_zv;                \
                         for (idx_t zv = start_zv; zv < stop_zv; zv++) { \
                             real_vec_t hval = gpd->readVecNorm(t, ARG_N(nv) \
                                                                xv, yv, zv, __LINE__); \
                             sendBuf->writeVecNorm(hval, index_nv,      \
                                                   index_xv, index_yv, izv++, __LINE__); \
                         } } while(0)
                         
                         // Include auto-generated loops to invoke calc_halo() from
                         // begin_*v to end_*v;
#include "stencil_halo_loops.hpp"
#undef calc_halo

                         // Send filled buffer to neighbor.
                         const void* buf = (const void*)(sendBuf->getRawData());
                         MPI_Isend(buf, sendBuf->get_num_bytes(), MPI_BYTE,
                                   neighbor_rank, int(gi), context.comm, &reqs[nreqs++]);
                         
                     }

                     // Receive data from same neighbor if buffer exists.
                     if (rcvBuf) {
                         void* buf = (void*)(rcvBuf->getRawData());
                         MPI_Irecv(buf, rcvBuf->get_num_bytes(), MPI_BYTE,
                                   neighbor_rank, int(gi), context.comm, &reqs[nreqs++]);
                     }
                     
                 } );

            // Wait for all to complete.
            // TODO: process each buffer asynchronously immediately upon completion.
            TRACE_MSG("rank %i: exchange_halos: waiting for %i MPI request(s)...",
                      context.my_rank, nreqs);
            MPI_Waitall(nreqs, reqs, MPI_STATUS_IGNORE);
            TRACE_MSG("rank %i: exchange_halos: done waiting for %i MPI request(s).",
                      context.my_rank, nreqs);

            // Unpack received data from all neighbors.
            context.bufs[gp].visitNeighbors
                (context,
                 [&](idx_t nn, idx_t nx, idx_t ny, idx_t nz,
                     int neighbor_rank,
                     Grid_NXYZ* sendBuf,
                     Grid_NXYZ* rcvBuf)
                 {
                     // Unpack data if buffer exists.
                     if (rcvBuf) {

                         // Set begin/end vars to indicate what part
                         // of main grid's halo to write to.
                         // Init range to whole rank size (inside halos).
                         idx_t begin_n = 0;
                         idx_t begin_x = 0;
                         idx_t begin_y = 0;
                         idx_t begin_z = 0;
                         idx_t end_n = context.dn;
                         idx_t end_x = context.dx;
                         idx_t end_y = context.dy;
                         idx_t end_z = context.dz;
                         
                         // Modify begin and/or end based on direction.
                         if (nn == idx_t(context.rank_prev)) { // neighbor is prev N.
                             begin_n = -hn; // begin at outside of halo.
                             end_n = 0;     // end at inside of halo.
                         }
                         if (nn == idx_t(context.rank_next)) { // neighbor is next N.
                             begin_n = context.dn; // begin at inside of halo.
                             end_n = context.dn + hn; // end of outside of halo.
                         }
                         if (nx == idx_t(context.rank_prev)) { // neighbor is on left.
                             begin_x = -hx;
                             end_x = 0;
                         }
                         if (nx == idx_t(context.rank_next)) { // neighbor is on right.
                             begin_x = context.dx;
                             end_x = context.dx + hx;
                         }
                         if (ny == idx_t(context.rank_prev)) { // neighbor is in front.
                             begin_y = -hy;
                             end_y = 0;
                         }
                         if (ny == idx_t(context.rank_next)) { // neighbor is in back.
                             begin_y = context.dy;
                             end_y = context.dy + hy;
                         }
                         if (nz == idx_t(context.rank_prev)) { // neighbor is above.
                             begin_z = -hz;
                             end_z = 0;
                         }
                         if (nz == idx_t(context.rank_next)) { // neighbor is below.
                             begin_z = context.dz;
                             end_z = context.dz + hz;
                         }

                         // Divide indices by vector lengths.
                         // Begin/end vars shouldn't be negative, so '/' is ok.
                         idx_t begin_nv = begin_n / VLEN_N;
                         idx_t begin_xv = begin_x / VLEN_X;
                         idx_t begin_yv = begin_y / VLEN_Y;
                         idx_t begin_zv = begin_z / VLEN_Z;
                         idx_t end_nv = end_n / VLEN_N;
                         idx_t end_xv = end_x / VLEN_X;
                         idx_t end_yv = end_y / VLEN_Y;
                         idx_t end_zv = end_z / VLEN_Z;

                         // TODO: fix this when MPI + wave-front is enabled.
                         idx_t t = begin_rt;
                         
                         // Define calc_halo to copy data from rcvBuf into main grid.
#define calc_halo(context, t,                                           \
                  start_nv, start_xv, start_yv, start_zv,               \
                  stop_nv, stop_xv, stop_yv, stop_zv)  do {             \
                             idx_t nv = start_nv;                       \
                             idx_t xv = start_xv;                       \
                             idx_t yv = start_yv;                       \
                             idx_t izv = index_zv * step_zv;            \
                             for (idx_t zv = start_zv; zv < stop_zv; zv++) { \
                                 real_vec_t hval =                      \
                                     rcvBuf->readVecNorm(index_nv,      \
                                                         index_xv, index_yv, izv++, __LINE__); \
                                 gpd->writeVecNorm(hval, t, ARG_N(nv)   \
                                                   xv, yv, zv, __LINE__); \
                     } } while(0)

                         // Include auto-generated loops to invoke calc_halo() from
                         // begin_*v to end_*v;
#include "stencil_halo_loops.hpp"
#undef calc_halo
                     }
                 } );

        } // grids.

        double end_time = getTimeInSecs();
        context.mpi_time += end_time - start_time;
#endif
    }
                         
            
    ///// StencilContext functions:

    // Init MPI-related vars.
    void StencilContext::setupMPI() {

        // Determine my position in 4D.
        Layout_4321 rank_layout(nrn, nrx, nry, nrz);
        idx_t mrnn, mrnx, mrny, mrnz;
        rank_layout.unlayout((idx_t)my_rank, mrnn, mrnx, mrny, mrnz);
        cout << "Logical coordinates of rank " << my_rank << ": " <<
            mrnn << ", " << mrnx << ", " << mrny << ", " << mrnz << endl;

        // Determine who my neighbors are.
        int num_neighbors = 0;
        for (int rn = 0; rn < num_ranks; rn++) {
            if (rn != my_rank) {
                idx_t rnn, rnx, rny, rnz;
                rank_layout.unlayout((idx_t)rn, rnn, rnx, rny, rnz);

                // Distance from me: prev => -1, self => 0, next => +1.
                idx_t rdn = rnn - mrnn;
                idx_t rdx = rnx - mrnx;
                idx_t rdy = rny - mrny;
                idx_t rdz = rnz - mrnz;

                // Rank rn is my neighbor if its distance <= 1 in every dim.
                if (abs(rdn) <= 1 && abs(rdx) <= 1 && abs(rdy) <= 1 && abs(rdz) <= 1) {

                    num_neighbors++;
                    cout << "Neighbor #" << num_neighbors << " at " <<
                        rnn << ", " << rnx << ", " << rny << ", " << rnz <<
                        " is rank " << rn << endl;
                    
                    // Size of buffer in each direction:
                    // if dist to neighbor is zero (i.e., is self), use full size,
                    // otherwise, use halo size.
                    idx_t rsn = (rdn == 0) ? dn : hn;
                    idx_t rsx = (rdx == 0) ? dx : hx;
                    idx_t rsy = (rdy == 0) ? dy : hy;
                    idx_t rsz = (rdz == 0) ? dz : hz;

                    // FIXME: only alloc buffers in directions actually needed, e.g.,
                    // many simple stencils don't need diagonals.
                    
                    // Is buffer needed?
                    if (rsn * rsx * rsy * rsz == 0) {
                        cout << "No halo exchange needed between ranks " << my_rank <<
                            " and " << rn << '.' << endl;
                    }

                    else {

                        // Add one to -1..+1 dist to get 0..2 range for my_neighbors indices.
                        rdn++; rdx++; rdy++; rdz++;

                        // Save rank of this neighbor.
                        my_neighbors[rdn][rdx][rdy][rdz] = rn;
                    
                        // Alloc MPI buffers between rn and me.
                        // Need send and receive for each updated grid.
                        for (auto gp : eqGridPtrs) {
                            for (int bd = 0; bd < Bufs::nBufDirs; bd++) {
                                ostringstream oss;
                                oss << gp->get_name();
                                if (bd == Bufs::bufSend)
                                    oss << "_send_halo_from_" << my_rank << "_to_" << rn;
                                else
                                    oss << "_get_halo_by_" << my_rank << "_from_" << rn;

                                bufs[gp].allocBuf(bd, rdn, rdx, rdy, rdz,
                                                  rsn, rsx, rsy, rsz,
                                                  oss.str());
                            }
                        }
                    }
                }
            }
        }
    }

    // Get total size.
    idx_t StencilContext::get_num_bytes() {
        idx_t nbytes = 0;
        for (auto gp : gridPtrs)
            nbytes += gp->get_num_bytes();
        for (auto pp : paramPtrs)
            nbytes += pp->get_num_bytes();
        for (auto gp : eqGridPtrs) {
            bufs[gp].visitNeighbors
                (*this,
                 [&](idx_t nn, idx_t nx, idx_t ny, idx_t nz,
                     int rank,
                     Grid_NXYZ* sendBuf,
                     Grid_NXYZ* rcvBuf)
                 {
                     if (sendBuf)
                         nbytes += sendBuf->get_num_bytes();
                     if (rcvBuf)
                         nbytes += rcvBuf->get_num_bytes();
                 } );
        }
        return nbytes;
    }

    // Init all grids & params w/same value within each,
    // but different values between them.
    void StencilContext::initSame() {
        real_t v = 0.1;
        cout << "Initializing grids..." << endl;
        for (auto gp : gridPtrs) {
            gp->set_same(v);
            v += 0.01;
        }
        if (paramPtrs.size()) {
            cout << "Initializing parameters..." << endl;
            for (auto pp : paramPtrs) {
                pp->set_same(v);
                v += 0.01;
            }
        }
    }

    // Init all grids & params w/different values.
    // Better for validation, but slower.
    void StencilContext::initDiff() {
        real_t v = 0.01;
        cout << "Initializing grids..." << endl;
        for (auto gp : gridPtrs) {
            gp->set_diff(v);
            v += 0.001;
        }
        if (paramPtrs.size()) {
            cout << "Initializing parameters..." << endl;
            for (auto pp : paramPtrs) {
                pp->set_diff(v);
                v += 0.001;
            }
        }
    }

    // Compare grids in contexts.
    // Params should not be written to, so they are not compared.
    // Return number of mis-compares.
    idx_t StencilContext::compare(const StencilContext& ref) const {

        cout << "Comparing grid(s) in '" << name << "' to '" << ref.name << "'..." << endl;
        if (gridPtrs.size() != ref.gridPtrs.size()) {
            cerr << "** number of grids not equal." << endl;
            return 1;
        }
        idx_t errs = 0;
        for (size_t gi = 0; gi < gridPtrs.size(); gi++) {
            cout << "Grid '" << ref.gridPtrs[gi]->get_name() << "'..." << endl;
            errs += gridPtrs[gi]->compare(*ref.gridPtrs[gi]);
        }

        cout << "Comparing parameter(s) in '" << name << "' to '" << ref.name << "'..." << endl;
        if (paramPtrs.size() != ref.paramPtrs.size()) {
            cerr << "** number of params not equal." << endl;
            return 1;
        }
        for (size_t pi = 0; pi < paramPtrs.size(); pi++) {
            errs += paramPtrs[pi]->compare(ref.paramPtrs[pi], EPSILON);
        }

        return errs;
    }
}
