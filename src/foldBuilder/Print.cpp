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

//////////// Implement methods for printing classes /////////////

#include "Print.hpp"
#include "CppIntrin.hpp"

////////////// Print visitors ///////////////

/////// Top-down

// A grid or parameter read.
// Uses the PrintHelper to format.
void PrintVisitorTopDown::visit(GridPoint* gp) {
    if (gp->isParam())
        _exprStr += _ph.readFromParam(_os, *gp);
    else
        _exprStr += _ph.readFromPoint(_os, *gp);
    _numCommon += _ph.getNumCommon(gp);
}

// An index.
void PrintVisitorTopDown::visit(IntTupleExpr* ite) {

    // get name of dimension, e.g., "x".
    assert(ite->getNumDims() == 1);
    _exprStr += ite->makeDimStr();
    _numCommon += _ph.getNumCommon(ite);
}

// An index expression.
void PrintVisitorTopDown::visit(IndexExpr* ie) {

    // E.g., "first_index(x)".
    _exprStr += ie->getFnName() + '(' + ie->getDirName() + ')';
    _numCommon += _ph.getNumCommon(ie);
}

// A constant.
// Uses the PrintHelper to format.
void PrintVisitorTopDown::visit(ConstExpr* ce) {
    _exprStr += _ph.addConstExpr(_os, ce->getNumVal());
    _numCommon += _ph.getNumCommon(ce);
}

// Some hand-written code.
// Uses the PrintHelper to format.
void PrintVisitorTopDown::visit(CodeExpr* ce) {
    _exprStr += _ph.addCodeExpr(_os, ce->getCode());
    _numCommon += _ph.getNumCommon(ce);
}

// Generic unary operators.
// Assumes unary operators have highest precedence, so no ()'s added.
void PrintVisitorTopDown::visit(UnaryNumExpr* ue) {
    _exprStr += ue->getOpStr();
    ue->getRhs()->accept(this);
    _numCommon += _ph.getNumCommon(ue);
}
void PrintVisitorTopDown::visit(UnaryBoolExpr* ue) {
    _exprStr += ue->getOpStr();
    ue->getRhs()->accept(this);
    _numCommon += _ph.getNumCommon(ue);
}
void PrintVisitorTopDown::visit(UnaryNum2BoolExpr* ue) {
    _exprStr += ue->getOpStr();
    ue->getRhs()->accept(this);
    _numCommon += _ph.getNumCommon(ue);
}

// Generic binary operators.
void PrintVisitorTopDown::visit(BinaryNumExpr* be) {
    _exprStr += "(";
    be->getLhs()->accept(this); // adds LHS to _exprStr.
    _exprStr += " " + be->getOpStr() + " ";
    be->getRhs()->accept(this); // adds RHS to _exprStr.
    _exprStr += ")";
    _numCommon += _ph.getNumCommon(be);
}
void PrintVisitorTopDown::visit(BinaryBoolExpr* be) {
    _exprStr += "(";
    be->getLhs()->accept(this); // adds LHS to _exprStr.
    _exprStr += " " + be->getOpStr() + " ";
    be->getRhs()->accept(this); // adds RHS to _exprStr.
    _exprStr += ")";
    _numCommon += _ph.getNumCommon(be);
}
void PrintVisitorTopDown::visit(BinaryNum2BoolExpr* be) {
    _exprStr += "(";
    be->getLhs()->accept(this); // adds LHS to _exprStr.
    _exprStr += " " + be->getOpStr() + " ";
    be->getRhs()->accept(this); // adds RHS to _exprStr.
    _exprStr += ")";
    _numCommon += _ph.getNumCommon(be);
}

// A commutative operator.
void PrintVisitorTopDown::visit(CommutativeExpr* ce) {
    _exprStr += "(";
    auto& ops = ce->getOps();
    int opNum = 0;
    for (auto ep : ops) {
        if (opNum > 0)
            _exprStr += " " + ce->getOpStr() + " ";
        ep->accept(this);   // adds operand to _exprStr;
        opNum++;
    }
    _exprStr += ")";
    _numCommon += _ph.getNumCommon(ce);
}

// A conditional operator.
void PrintVisitorTopDown::visit(IfExpr* ie) {

    // Null ptr => no condition.
    if (ie->getCond()) {
        ie->getCond()->accept(this); // sets _exprStr;
        string cond = getExprStrAndClear();

        // pseudo-code format.
        _os << _ph.getLinePrefix() << "IF (" << cond << ") THEN" << endl;
    }
    
    // Get assignment expr and clear expr.
    ie->getExpr()->accept(this); // writes to _exprStr;
    string vexpr = getExprStrAndClear();

    // note: _exprStr is now empty.
    // note: no need to update num-common.
}

// An equals operator.
void PrintVisitorTopDown::visit(EqualsExpr* ee) {

    // Get RHS and clear expr.
    ee->getRhs()->accept(this); // writes to _exprStr;
    string rhs = getExprStrAndClear();

    // Write statement with embedded rhs.
    GridPointPtr gpp = ee->getLhs();
    _os << _ph.getLinePrefix() << _ph.writeToPoint(_os, *gpp, rhs) << _ph.getLineSuffix();

    // note: _exprStr is now empty.
    // note: no need to update num-common.
}

/////// Bottom-up

// If 'comment' is set, use it for the comment.
// Return stream to continue w/RHS.
ostream& PrintVisitorBottomUp::makeNextTempVar(Expr* ex, string comment) {
    _exprStr = _ph.makeVarName();
    if (ex) {
        _tempVars[ex] = _exprStr;
        if (comment.length() == 0)
            _os << endl << " // " << _exprStr << " = " << ex->makeStr() << "." << endl;
    }
    if (comment.length())
        _os << endl << " // " << _exprStr << " = " << comment << "." << endl;
    _os << _ph.getLinePrefix() << _ph.getVarType() << " " << _exprStr << " = ";
    return _os;
}

// Try some simple printing techniques.
// Return true if printing is done.
// Return false if more complex method should be used.
// TODO: the current code causes all nodes in an expr above a certain
// point to avoid top-down printing because it looks at the original expr,
// not the new one with temp vars. Fix this.
bool PrintVisitorBottomUp::trySimplePrint(Expr* ex, bool force) {

    bool exprDone = false;

    // How many nodes in ex?
    int exprSize = ex->getNumNodes();
    bool tooBig = exprSize > _maxExprSize;
    bool tooSmall = exprSize < _minExprSize;

    // Determine whether this expr has already been evaluated
    // and a variable holds its result.
    auto p = _tempVars.find(ex);
    if (p != _tempVars.end()) {

        // if so, just use the existing var.
        _exprStr = p->second;
        exprDone = true;
    }
        
    // Consider top down if forcing or expr <= maxExprSize.
    else if (force || !tooBig) {

        // use a top-down printer to render the expr.
        PrintVisitorTopDown* topDown = newPrintVisitorTopDown();
        ex->accept(topDown);

        // were there any common subexprs found?
        int numCommon = topDown->getNumCommon();

        // if no common subexprs, use the top-down expression.
        if (numCommon == 0) {
            _exprStr = topDown->getExprStr();
            exprDone = true;
        }
            
        // if common subexprs exist, and top-down is forced, use the
        // top-down expression regardless.  If node is big enough for
        // sharing, also assign the result to a temp var so it can be used
        // later.
        else if (force) {
            if (tooSmall)
                _exprStr = topDown->getExprStr();
            else
                makeNextTempVar(ex) << topDown->getExprStr() << _ph.getLineSuffix();
            exprDone = true;
        }

        // otherwise, there are common subexprs, and top-down is not forced,
        // so don't do top-down.
        
        delete topDown;
    }

    if (force) assert(exprDone);
    return exprDone;
}

// A grid or param point: just set expr.
void PrintVisitorBottomUp::visit(GridPoint* gp) {
    trySimplePrint(gp, true);
}

// An index.
void PrintVisitorBottomUp::visit(IntTupleExpr* ite) {
    trySimplePrint(ite, true);
}

// A constant: just set expr.
void PrintVisitorBottomUp::visit(ConstExpr* ce) {
    trySimplePrint(ce, true);
}

// Code: just set expr.
void PrintVisitorBottomUp::visit(CodeExpr* ce) {
    trySimplePrint(ce, true);
}

// A numerical unary operator.
void PrintVisitorBottomUp::visit(UnaryNumExpr* ue) {

    // Try top-down on whole expression.
    // Example: '-a' creates no immediate output,
    // and '-a' is saved in _exprStr.
    if (trySimplePrint(ue, false))
        return;

    // Expand the RHS, then apply operator to result.
    // Example: '-(a * b)' might output the following:
    // temp1 = a * b;
    // temp2 = -temp1;
    // with 'temp2' saved in _exprStr.
    ue->getRhs()->accept(this); // sets _exprStr.
    string rhs = getExprStrAndClear();
    makeNextTempVar(ue) << ue->getOpStr() << ' ' << rhs << _ph.getLineSuffix();
}

// A numerical binary operator.
void PrintVisitorBottomUp::visit(BinaryNumExpr* be) {

    // Try top-down on whole expression.
    // Example: 'a/b' creates no immediate output,
    // and 'a/b' is saved in _exprStr.
    if (trySimplePrint(be, false))
        return;

    // Expand both sides, then apply operator to result.
    // Example: '(a * b) / (c * d)' might output the following:
    // temp1 = a * b;
    // temp2 = b * c;
    // temp3 = temp1 / temp2;
    // with 'temp3' saved in _exprStr.
    be->getLhs()->accept(this); // sets _exprStr.
    string lhs = getExprStrAndClear();
    be->getRhs()->accept(this); // sets _exprStr.
    string rhs = getExprStrAndClear();
    makeNextTempVar(be) << lhs << ' ' << be->getOpStr() << ' ' << rhs << _ph.getLineSuffix();
}

// Boolean unary and binary operators.
// For now, don't try to use bottom-up for these.
// TODO: investigate whether there is any potential
// benefit in doing this.
void PrintVisitorBottomUp::visit(UnaryBoolExpr* ue) {
    trySimplePrint(ue, true);
}
void PrintVisitorBottomUp::visit(BinaryBoolExpr* be) {
    trySimplePrint(be, true);
}
void PrintVisitorBottomUp::visit(BinaryNum2BoolExpr* be) {
    trySimplePrint(be, true);
}

// A commutative operator.
void PrintVisitorBottomUp::visit(CommutativeExpr* ce) {

    // Try top-down on whole expression.
    // Example: 'a*b' creates no immediate output,
    // and 'a*b' is saved in _exprStr.
    if (trySimplePrint(ce, false))
        return;

    // Make separate assignment for N-1 operands.
    // Example: 'a + b + c + d' might output the following:
    // temp1 = a + b;
    // temp2 = temp1 + c;
    // temp3 = temp2 = d;
    // with 'temp3' left in _exprStr;
    auto& ops = ce->getOps();
    assert(ops.size() > 1);
    string lhs, exStr;
    int opNum = 0;
    for (auto ep : ops) {
        opNum++;

        // eval the operand; sets _exprStr.            
        ep->accept(this);
        string opStr = getExprStrAndClear();

        // first operand; just save as LHS for next iteration.
        if (opNum == 1) {
            lhs = opStr;
            exStr = ep->makeStr();
        }

        // subsequent operands.
        // makes separate assignment for each one.
        // result is kept as LHS of next one.
        else {

            // Use whole expression only for the last step.
            Expr* ex = (opNum == (int)ops.size()) ? ce : NULL;

            // Add RHS to partial-result comment.
            exStr += ' ' + ce->getOpStr() + ' ' + ep->makeStr();

            // Output this step.
            makeNextTempVar(ex, exStr) << lhs << ' ' << ce->getOpStr() << ' ' <<
                opStr << _ph.getLineSuffix();
            lhs = getExprStr(); // result used in next iteration, if any.
        }
    }

    // note: _exprStr contains result of last operation.
}

// Conditional.
void PrintVisitorBottomUp::visit(IfExpr* ie) {
    trySimplePrint(ie, true);
    // note: _exprStr is now empty.
}

// An equality.
void PrintVisitorBottomUp::visit(EqualsExpr* ee) {

    // Note that we don't try top-down here.
    // We always assign the RHS to a temp var and then
    // write the temp var to the grid.
    
    // Eval RHS.
    Expr* rp = ee->getRhs().get();
    rp->accept(this); // sets _exprStr.
    string rhs = _exprStr;

    // Assign RHS to a temp var.
    makeNextTempVar(rp) << rhs << _ph.getLineSuffix(); // sets _exprStr.
    string tmp = getExprStrAndClear();

    // Write temp var to grid.
    GridPointPtr gpp = ee->getLhs();
    _os << "\n // Define value at " << gpp->makeStr() << ".\n";
    _os << _ph.getLinePrefix() << _ph.writeToPoint(_os, *gpp, tmp) << _ph.getLineSuffix();

    // note: _exprStr is now empty.
}

///////// POVRay.

// Only want to visit the RHS of an equality.
void POVRayPrintVisitor::visit(EqualsExpr* ee) {
    ee->getRhs()->accept(this);      
}
    
// A point: output it.
void POVRayPrintVisitor::visit(GridPoint* gp) {
    _numPts++;

    // Pick a color based on its distance.
    size_t ci = gp->max();
    ci %= _colors.size();
        
    _os << "point(" + _colors[ci] + ", " << gp->makeValStr() << ")" << endl;
}

////// DOT-language.

// A grid or parameter access.
void DOTPrintVisitor::visit(GridPoint* gp) {
    string label = getLabel(gp);
    if (label.size())
        _os << label << " [ shape = box ];" << endl;
}

// A constant.
// TODO: don't share node.
void DOTPrintVisitor::visit(ConstExpr* ce) {
    string label = getLabel(ce);
    if (label.size())
        _os << label << endl;
}

// Some hand-written code.
void DOTPrintVisitor::visit(CodeExpr* ce) {
    string label = getLabel(ce);
    if (label.size())
        _os << label << endl;
}

// Generic numeric unary operators.
void DOTPrintVisitor::visit(UnaryNumExpr* ue) {
    string label = getLabel(ue);
    if (label.size())
        _os << label << " [ label = \"" << ue->getOpStr() << "\" ];" << endl;
    _os << ue->makeQuotedStr() << " -> " << ue->getRhs()->makeQuotedStr() << ";" << endl;
    ue->getRhs()->accept(this);
}

// Generic numeric binary operators.
void DOTPrintVisitor::visit(BinaryNumExpr* be) {
    string label = getLabel(be);
    if (label.size())
        _os << label << " [ label = \"" << be->getOpStr() << "\" ];" << endl;
    _os << be->makeQuotedStr() << " -> " << be->getLhs()->makeQuotedStr() << ";" << endl <<
        be->makeQuotedStr() << " -> " << be->getRhs()->makeQuotedStr() << ";" << endl;
    be->getLhs()->accept(this);
    be->getRhs()->accept(this);
}

// A commutative operator.
void DOTPrintVisitor::visit(CommutativeExpr* ce) {
    string label = getLabel(ce);
    if (label.size())
        _os << label << " [ label = \"" << ce->getOpStr() << "\" ];" << endl;
    for (auto ep : ce->getOps()) {
        _os << ce->makeQuotedStr() << " -> " << ep->makeQuotedStr() << ";" << endl;
        ep->accept(this);
    }
}

// An equals operator.
void DOTPrintVisitor::visit(EqualsExpr* ee) {
    string label = getLabel(ee);
    if (label.size())
        _os << label << " [ label = \"==\" ];" << endl;
    _os << ee->makeQuotedStr() << " -> " << ee->getLhs()->makeQuotedStr()  << ";" << endl <<
        ee->makeQuotedStr() << " -> " << ee->getRhs()->makeQuotedStr() << ";" << endl;
    ee->getLhs()->accept(this);
    ee->getRhs()->accept(this);
}

// A grid or parameter access.
void SimpleDOTPrintVisitor::visit(GridPoint* gp) {
    if (gp->isParam())
        return;
    string label = getLabel(gp);
    if (label.size())
        _os << label << " [ shape = box ];" << endl;
    _gridsSeen.insert(gp->makeQuotedStr());
}

// Generic numeric unary operators.
void SimpleDOTPrintVisitor::visit(UnaryNumExpr* ue) {
    ue->getRhs()->accept(this);
}

// Generic numeric binary operators.
void SimpleDOTPrintVisitor::visit(BinaryNumExpr* be) {
    be->getLhs()->accept(this);
    be->getRhs()->accept(this);
}

// A commutative operator.
void SimpleDOTPrintVisitor::visit(CommutativeExpr* ce) {
    for (auto ep : ce->getOps())
        ep->accept(this);
}

// An equals operator.
void SimpleDOTPrintVisitor::visit(EqualsExpr* ee) {

    // LHS is source.
    ee->getLhs()->accept(this);
    string label = ee->makeQuotedStr();
    for (auto g : _gridsSeen)
        label = g;              // really should only be one.
    _gridsSeen.clear();

    // RHS nodes are target.
    ee->getRhs()->accept(this);
    for (auto g : _gridsSeen)
        _os << label << " -> " << g  << ";" << endl;
    _gridsSeen.clear();
}

////////////// Printers ///////////////

/////// Pseudo-code.

// Print out a stencil in human-readable form, for debug or documentation.
void PseudoPrinter::print(ostream& os) {

    os << "Stencil '" << _stencil.getName() << "' pseudo-code:" << endl;

    // Loop through all eqGroups.
    for (auto& eq : _eqGroups) {

        string eqName = eq.getName();
        os << endl << " ////// Equation group '" << eqName <<
            "' //////" << endl;

        CounterVisitor cv;
        eq.visitEqs(&cv);
        PrintHelper ph(&cv, "temp", "real", " ", ".\n");

        if (eq.cond.get()) {
            string condStr = eq.cond->makeStr();
            os << endl << " // Valid under the following condition:" << endl <<
                ph.getLinePrefix() << "IF " << condStr << ph.getLineSuffix();
        }
        else
            os << endl << " // Valid under the default condition." << endl;

        os << endl << " // Top-down stencil calculation:" << endl;
        PrintVisitorTopDown pv1(os, ph);
        eq.visitEqs(&pv1);
            
        os << endl << " // Bottom-up stencil calculation:" << endl;
        PrintVisitorBottomUp pv2(os, ph, _maxExprSize, _minExprSize);
        eq.visitEqs(&pv2);
    }
}

///// DOT language.

// Print out a stencil in DOT form
void DOTPrinter::print(ostream& os) {

    DOTPrintVisitor* pv = _isSimple ?
        new SimpleDOTPrintVisitor(os) :
        new DOTPrintVisitor(os);

    os << "digraph \"Stencil " << _stencil.getName() << "\" {" << endl;

    // Loop through all eqGroups.
    for (auto& eq : _eqGroups) {
        //os << "subgraph \"Equation-group " << eq.getName() << "\" {" << endl;
        eq.visitEqs(pv);
        //os << "}" << endl;
    }
    os << "}" << endl;
    delete pv;
}


///// POVRay.

// Print out a stencil in POVRay form.
void POVRayPrinter::print(ostream& os) {

    os << "#include \"stencil.inc\"" << endl;
    int cpos = 25;
    os << "camera { location <" <<
        cpos << ", " << cpos << ", " << cpos << ">" << endl <<
        "  look_at <0, 0, 0>" << endl <<
        "}" << endl;

    // Loop through all eqGroups.
    for (auto& eq : _eqGroups) {

        // TODO: separate mutiple grids.
        POVRayPrintVisitor pv(os);
        eq.visitEqs(&pv);
        os << " // " << pv.getNumPoints() << " stencil points" << endl;
    }
}

///// YASK.

// Print YASK code in new stencil context class.
// TODO: split this into smaller methods.
void YASKCppPrinter::printCode(ostream& os) {
    map<Grid*, string> typeNames, dimArgs, haloArgs, padArgs, ofsArgs;
    map<Param*, string> paramTypeNames, paramDimArgs;

    os << "// Automatically generated code; do not edit." << endl;

    os << endl << "////// YASK implementation of the '" << _stencil.getName() <<
        "' stencil //////" << endl;
    os << endl << "namespace yask {" << endl;

    // Create the overall context base.
    {
        // get stats.
        CounterVisitor cve;
        _eqGroups.visitEqs(&cve);

        // TODO: get rid of global max-halo concept by using grid-specific halos.
        IntTuple maxHalos;      

        os << endl << " ////// Stencil-specific data //////" << endl <<
            "struct " << _context_base << " : public StencilContext {" << endl;

        // Grids.
        os << endl << " // Grids." << endl;
        for (auto gp : _grids) {
            assert (!gp->isParam());
            string grid = gp->getName();

            // Type name & ctor params.
            // Name in kernel is 'Grid_' followed by dimensions.
            string typeName = "Grid_";
            string templStr, dimArg, haloArg, padArg, ofsArg;
            for (auto dim : gp->getDims()) {
                string ucDim = allCaps(dim);
                typeName += ucDim;

                // step dimension.
                if (dim == _dims._stepDim) {
                    string sdvar = grid + "_alloc_" + dim;
                    os << " static const idx_t " << sdvar << " = " <<
                        gp->getStepDimSize() << ";\n";
                    templStr = "<" + sdvar + ">";
                }
                
                // non-step dimension.
                else {

                    // Halo for this dimension.
                    int halo = _settings._haloSize > 0 ?
                        _settings._haloSize : gp->getHaloSize(dim);
                    string hvar = grid + "_halo_" + dim;
                    os << " const idx_t " << hvar << " = " << halo << ";\n";

                    // Update max halo across grids.
                    int* mh = maxHalos.lookup(dim);
                    if (mh)
                        *mh = max(*mh, halo);
                    else
                        maxHalos.addDimBack(dim, halo);

                    // Ctor args.
                    dimArg += "_opts->d" + dim + ", ";
                    haloArg += hvar + ", ";
                    padArg += "_opts->p" + dim + ", ";
                    ofsArg += "ofs_" + dim + ", ";
                }
            }
            typeName += templStr;
            typeNames[gp] = typeName;
            dimArgs[gp] = dimArg;
            haloArgs[gp] = haloArg;
            padArgs[gp] = padArg;
            ofsArgs[gp] = ofsArg;
            os << " " << typeName << "* " << grid << "; // ";
            if (_eqGroups.getOutputGrids().count(gp) == 0)
                os << "not ";
            os << "updated by stencil." << endl;
        }

        // Max halos.
        os << endl << " // Max halos across all grids." << endl;
        for (auto dim : maxHalos.getDims())
            os << " const idx_t max_halo_" << dim << " = " <<
                maxHalos.getVal(dim) << ";" << endl;

        // Parameters.
        if (_params.size())
            os << endl << " // Parameters." << endl;
        for (auto pp : _params) {
            assert(pp->isParam());
            string param = pp->getName();

            // Type name.
            // Name in kernel is 'GenericGridNd<real_t>'.
            ostringstream oss;
            oss << "GenericGrid" << pp->size() << "d<real_t";
            if (pp->size()) {
                oss << ",Layout_";
                for (int dn = pp->size(); dn > 0; dn--)
                    oss << dn;
            }
            oss << ">";
            string typeName = oss.str();

            // Ctor params.
            string dimArg = pp->makeValStr();
            
            paramTypeNames[pp] = typeName;
            paramDimArgs[pp] = dimArg;
            os << " " << typeName << "* " << param << ";" << endl;
        }

        // Ctor.
        {
            os << endl <<
                " // Constructor.\n" <<
                " " << _context_base << "(StencilSettings& settings) :"
                " StencilContext(settings) {\n" <<
                "  name = \"" << _stencil.getName() << "\";\n";
            
            // Init grid ptrs.
            for (auto gp : _grids) {
                string grid = gp->getName();
                os << " " << grid << " = 0;" << endl;
                os << " gridNames.insert(\"" << grid << "\");" << endl;

                // Output grids.
                if (_eqGroups.getOutputGrids().count(gp))
                    os << " outputGridNames.insert(\"" << grid << "\");" << endl;
            }
            
            // Init param ptrs.
            for (auto pp : _params) {
                string param = pp->getName();
                os << " " << param << " = 0;" << endl;
                os << " paramNames.insert(\"" << param << "\");" << endl;
            }

            // Init halo sizes.
            for (auto dim : maxHalos.getDims())
                os << " h" << dim << " = ROUND_UP(max_halo_" << dim <<
                    ", VLEN_" << allCaps(dim) << ");" << endl;
            
            // end of ctor.
            os << " }" << endl;
        }
        os << "}; // " << _context_base << endl;
    }
        
    // A struct for each eqGroup.
    for (size_t ei = 0; ei < _eqGroups.size(); ei++) {

        // Scalar eqGroup.
        auto& eq = _eqGroups.at(ei);
        string eqName = eq.getName();
        string eqDesc = eq.getDescription();
        string egsName = "EqGroup_" + eqName;

        os << endl << " ////// Stencil " << eqDesc << " //////\n" <<
            "\n struct " << egsName << " {\n" <<
            " std::string name = \"" << eqName << "\";\n";

        // Ops for this eqGroup.
        CounterVisitor fpops;
        eq.visitEqs(&fpops);
            
        // Example computation.
        os << endl << " // " << fpops.getNumOps() << " FP operation(s) per point:" << endl;
        addComment(os, eq);
        os << " const int scalar_fp_ops = " << fpops.getNumOps() << ";" << endl <<
            " const int scalar_points_updated = " << eq.getNumEqs() << ";" << endl;

        // Init code.
        {
            os << " void setPtrs(" << _context_base << "& context, "
                "GridPtrs& outputGridPtrs, GridPtrs& inputGridPtrs) {" << endl;

            // I/O grids.
            if (eq.getOutputGrids().size()) {
                os << "\n // The following grids are written by " << egsName << endl;
                for (auto gp : eq.getOutputGrids())
                    os << "  outputGridPtrs.push_back(context." << gp->getName() << ");" << endl;
            }
            if (eq.getInputGrids().size()) {
                os << "\n // The following grids are read by " << egsName << endl;
                for (auto gp : eq.getInputGrids())
                    if (!gp->isParam())
                        os << "  inputGridPtrs.push_back(context." << gp->getName() << ");" << endl;
            }
            os << " } // setPtrs." << endl;
        }

        // Condition.
        {
            os << endl << " // Determine whether " << egsName << " is valid at the given indices. " <<
                "Return true if indices are within the valid sub-domain or false otherwise." << endl <<
                    " bool is_in_valid_domain(" << _context_base << "& context, " <<
                _dims._allDims.makeDimStr(", ", "idx_t ") << ") {" << endl;
            if (eq.cond.get())
                os << " return " << eq.cond->makeStr() << ";" << endl;
            else
                os << " return true; // full domain." << endl;
            os << " }" << endl;
        }
        
        // Scalar code.
        {
            // C++ scalar print assistant.
            CounterVisitor cv;
            eq.visitEqs(&cv);
            CppPrintHelper* sp = new CppPrintHelper(&cv, "temp", "real_t", " ", ";\n");
            
            // Stencil-calculation code.
            // Function header.
            os << endl << " // Calculate one scalar result relative to indices " <<
                _dims._allDims.makeDimStr(", ") << "." << endl;
            os << " void calc_scalar(" << _context_base << "& context, " <<
                _dims._allDims.makeDimStr(", ", "idx_t ") << ") {" << endl;

            // C++ code generator.
            // The visitor is accepted at all nodes in the scalar AST;
            // for each node in the AST, code is generated and
            // stored in the expression-string in the visitor.
            PrintVisitorBottomUp pcv(os, *sp, _maxExprSize, _minExprSize);

            // Generate the code.
            eq.visitEqs(&pcv);

            // End of function.
            os << "} // calc_scalar." << endl;

            delete sp;
        }

        // Cluster/Vector code.
        {
            // Cluster eqGroup at same index.
            // This should be the same eq-group because it was copied from the
            // scalar one.
            auto& ceq = _clusterEqGroups.at(ei);
            assert(eqDesc == ceq.getDescription());

            // Create vector info for this eqGroup.
            // The visitor is accepted at all nodes in the cluster AST;
            // for each grid access node in the AST, the vectors
            // needed are determined and saved in the visitor.
            VecInfoVisitor vv(_dims);
            ceq.visitEqs(&vv);

            // Reorder based on vector info.
            ExprReorderVisitor erv(vv);
            ceq.visitEqs(&erv);
            
            // C++ vector print assistant.
            CounterVisitor cv;
            ceq.visitEqs(&cv);
            CppVecPrintHelper* vp = newPrintHelper(vv, cv);
            
            // Stencil-calculation code.
            // Function header.
            int numResults = _dims._clusterPts.product();
            os << endl << " // Calculate " << numResults <<
                " result(s) relative to indices " << _dims._allDims.makeDimStr(", ") <<
                " in a '" << _dims._clusterPts.makeDimValStr(" * ") <<
                "' cluster containing " << _dims._clusterMults.product() << " '" <<
                _dims._fold.makeDimValStr(" * ") << "' vector(s)." << endl;
            os << " // Indices must be normalized, i.e., already divided by VLEN_*." << endl;
            os << " // SIMD calculations use " << vv.getNumPoints() <<
                " vector block(s) created from " << vv.getNumAlignedVecs() <<
                " aligned vector-block(s)." << endl;
            os << " // There are approximately " << (fpops.getNumOps() * numResults) <<
                " FP operation(s) per invocation." << endl;
            os << " void calc_cluster(" << _context_base << "& context, " <<
                _dims._allDims.makeDimStr(", ", "idx_t ", "v") << ") {" << endl;

            // Element indices.
            os << endl << " // Element (un-normalized) indices." << endl;
            for (auto dim : _dims._allDims.getDims()) {
                auto p = _dims._fold.lookup(dim);
                os << " idx_t " << dim << " = " << dim << "v";
                if (p) os << " * " << *p;
                os << ";" << endl;
            }
                
            // Code generator visitor.
            // The visitor is accepted at all nodes in the cluster AST;
            // for each node in the AST, code is generated and
            // stored in the expression-string in the visitor.
            PrintVisitorBottomUp pcv(os, *vp, _maxExprSize, _minExprSize);

            // Generate the code.
            // Visit all expressions to cover the whole cluster.
            ceq.visitEqs(&pcv);

            // End of function.
            os << "} // calc_cluster." << endl;

            // Generate prefetch code for no specific direction and then each
            // orthogonal direction.
            for (int diri = -1; diri < _dims._allDims.size(); diri++) {

                // Create a direction object.
                // If diri < 0, there is no direction.
                // If diri >= 0, add a direction.
                IntTuple dir;
                if (diri >= 0) {
                    string dim = _dims._allDims.getDims()[diri];

                    // Magnitude of dimension is based on cluster.
                    const int* p = _dims._clusterMults.lookup(dim);
                    int m = p ? *p : 1;
                    dir.addDimBack(dim, m);

                    // Don't need prefetch in step dimension.
                    if (dim == _dims._stepDim)
                        continue;
                }

                // Function header.
                os << endl << " // Prefetches cache line(s) ";
                if (dir.size())
                    os << "for leading edge of stencil advancing by " <<
                        dir.getDirVal() << " vector(s) in '+" <<
                        dir.getDirName() << "' direction ";
                else
                    os << "for entire stencil ";
                os << "relative to indices " << _dims._allDims.makeDimStr(", ") <<
                    " in a '" << _dims._clusterPts.makeDimValStr(" * ") <<
                    "' cluster containing " << _dims._clusterMults.product() << " '" <<
                    _dims._fold.makeDimValStr(" * ") << "' vector(s)." << endl;
                os << " // Indices must be normalized, i.e., already divided by VLEN_*." << endl;

                string fname = "prefetch_cluster";
                if (dir.size())
                    fname += "_" + dir.getDirName();
                os << " template<int level> void " << fname <<
                    "(" << _context_base << "& context, " <<
                    _dims._allDims.makeDimStr(", ", "idx_t ", "v") << ") {" << endl;

                // C++ prefetch code.
                vp->printPrefetches(os, dir);

                // End of function.
                os << "} // " << fname << "." << endl;

            } // direction.

            delete vp;
        }

        os << "};" << endl; // end of class.
            
    } // stencil eqGroups.

    // Finish the context.
    {
        os << endl << " ////// Overall stencil-specific context //////" << endl <<
            "struct " << _context << " : public " << _context_base << " {" << endl;

        // Stencil eqGroup objects.
        os << endl << " // Stencil equation-groups." << endl;
        for (auto& eg : _eqGroups) {
            string egName = eg.getName();
            os << " EqGroupTemplate<EqGroup_" << egName << "," <<
                _context << "> eqGroup_" << egName << ";" << endl;
        }

        // Ctor.
        os << endl <<
            " // Constructor.\n" <<
            " " << _context << "(StencilSettings& settings) : " <<
            _context_base << "(settings) { setPtrs(); }" << endl;
        
        // Init code.
        {
            os << endl <<
                " // Set generic pointers to stencil-specific objects.\n"
                " virtual void setPtrs() {\n";
            
            // Push eq-group pointers to list.
            os << "\n // Equation groups.\n"
                "  eqGroups.clear();\n";
            for (auto& eg : _eqGroups) {
                string egName = eg.getName();
                os << "  eqGroups.push_back(&eqGroup_" << egName << ");\n";

                // Add dependencies.
                for (DepType dt = certain_dep; dt < num_deps; dt = DepType(dt+1)) {
                    for (auto dep : eg.getDeps(dt)) {
                        string dtName = (dt == certain_dep) ? "certain_dep" :
                            (dt == possible_dep) ? "possible_dep" :
                            "internal_error";
                        os << "  eqGroup_" << egName <<
                            ".add_dep(yask::" << dtName <<
                            ", &eqGroup_" << dep << ");\n";
                    }
                }
            }

            // Push grid pointers to list.
            os << "\n // Grids.\n"
                "  gridPtrs.clear();\n"
                "  outputGridPtrs.clear();\n";
            for (auto gp : _grids) {
                string grid = gp->getName();
                os << "  gridPtrs.push_back(" << grid << ");" << endl;
                
                // I/O grids.
                if (_eqGroups.getOutputGrids().count(gp))
                    os << "  outputGridPtrs.push_back(" << grid  << ");" << endl;
            }

            // Push param pointers to list.
            os << "\n // Parameters.\n"
                "  paramPtrs.clear();\n";
            for (auto pp : _params) {
                string param = pp->getName();
                os << "  paramPtrs.push_back(" << param << ");" << endl;
            }
            
            os << " }" << endl;
        }
        
        // Allocate grids.
        {
            os << endl << " virtual void allocGrids() {" << endl;
            for (auto gp : _grids) {
                string grid = gp->getName();
                string hbwArg = "false";
                if ((_eqGroups.getOutputGrids().count(gp) && _settings._hbwRW) ||
                    (_eqGroups.getOutputGrids().count(gp) == 0 && _settings._hbwRO))
                    hbwArg = "true";
                os << "  " << grid << " = new " << typeNames[gp] <<
                    "(" << dimArgs[gp] << haloArgs[gp] << padArgs[gp] << ofsArgs[gp] <<
                    "\"" << grid << "\", " <<
                    hbwArg << ", get_ostr());\n";
            }
            os << " }" << endl;
        }
        
        // Allocate params.
        {
            os << endl << " virtual void allocParams() {" << endl;
            for (auto pp : _params) {
                string param = pp->getName();
                os << "  " << param << " = new " << paramTypeNames[pp] <<
                    "(" << paramDimArgs[pp] << ");\n";
            }
            os << " }" << endl;
        }
        
        // Stencil provided code for StencilContext
        CodeList *extraCode;
        if ( (extraCode = _stencil.getExtensionCode(STENCIL_CONTEXT)) != NULL )
        {
            os << endl << "  // Functions provided by user" << endl;
            for ( auto code : *extraCode )
                os << code << endl;
        }
        
        os << "}; // " << _context << endl;
    }
    os << "} // namespace yask." << endl;
        
}

// Print YASK grids.
// Under development--not used.
void YASKCppPrinter::printGrids(ostream& os) {
    os << "// Automatically generated code; do not edit." << endl;

    os << endl << "////// Grid classes needed to implement the '" << _stencil.getName() <<
        "' stencil //////" << endl;
    os << endl << "namespace yask {" << endl;
    
    os << "} // namespace yask." << endl;
}

// Print YASK macros.  TODO: many hacks below assume certain dimensions and
// usage model by the kernel. Need to improve kernel to make it more
// flexible and then communicate info more generically. Goal is to get rid
// of all these macros.
void YASKCppPrinter::printMacros(ostream& os) {
    os << "// Automatically generated code; do not edit." << endl;

    os << endl;
    os << "// Stencil:" << endl;
    os << "#define YASK_STENCIL_NAME \"" << _stencil.getName() << "\"" << endl;
    os << "#define YASK_STENCIL_IS_" << allCaps(_stencil.getName()) << " (1)" << endl;
    os << "#define YASK_STENCIL_CONTEXT " << _context << endl;

    os << endl;
    os << "// Dimensions:" << endl;
    for (auto dim : _dims._allDims.getDims()) {
        os << "#define USING_DIM_" << allCaps(dim) << " (1)" << endl;
    }
        
    // Vec/cluster lengths.
    os << endl;
    os << "// One vector fold: " << _dims._fold.makeDimValStr(" * ") << endl;
    for (auto dim : _dims._fold.getDims()) {
        string ucDim = allCaps(dim);
        os << "#define VLEN_" << ucDim << " (" << _dims._fold.getVal(dim) << ")" << endl;
    }
    os << "#define VLEN (" << _dims._fold.product() << ")" << endl;
    os << "#define VLEN_FIRST_DIM_IS_UNIT_STRIDE (" <<
        (IntTuple::getDefaultFirstInner() ? 1 : 0) << ")" << endl;
    os << "#define USING_UNALIGNED_LOADS (" <<
        (_settings._allowUnalignedLoads ? 1 : 0) << ")" << endl;

    os << endl;
    os << "// Cluster multipliers of vector folds: " <<
        _dims._clusterMults.makeDimValStr(" * ") << endl;
    for (auto dim : _dims._clusterMults.getDims()) {
        string ucDim = allCaps(dim);
        os << "#define CLEN_" << ucDim << " (" <<
            _dims._clusterMults.getVal(dim) << ")" << endl;
    }
    os << "#define CLEN (" << _dims._clusterMults.product() << ")" << endl;
}
