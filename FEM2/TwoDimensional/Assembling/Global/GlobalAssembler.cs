using FEM2.Core;
using FEM2.Core.Base;
using FEM2.Core.Boundary;
using FEM2.Core.Global;
using FEM2.Core.GridComponents;
using FEM2.FEM.Assembling;
using FEM2.FEM.Assembling.Local;
using FEM2.TwoDimensional.Assembling.Local;
using System.Diagnostics;

namespace FEM2.TwoDimensional.Assembling.Global;

public class GlobalAssembler<TNode>
{
    private readonly Grid<Node2D> _grid;
    private readonly IMatrixPortraitBuilder<TNode, SymmetricSparseMatrix> _matrixPortraitBuilder;
    private readonly ILocalAssembler _localAssembler;
    private readonly IInserter<SymmetricSparseMatrix> _inserter;
    private readonly GaussExcluder _gaussExсluder;
    private Equation<SymmetricSparseMatrix> _equation;
    private SymmetricSparseMatrix _preconditionMatrix;

    public GlobalAssembler
    (
        Grid<Node2D> grid,
        IMatrixPortraitBuilder<TNode, SymmetricSparseMatrix> matrixPortraitBuilder,
        ILocalAssembler localAssembler,
        IInserter<SymmetricSparseMatrix> inserter,
        GaussExcluder gaussExсluder
    )
    {
        _grid = grid;
        _matrixPortraitBuilder = matrixPortraitBuilder;
        _localAssembler = localAssembler;
        _inserter = inserter;
        _gaussExсluder = gaussExсluder;
    }

    public GlobalAssembler<TNode> AssembleEquation(Grid<TNode> grid)
    {
        var globalMatrix = _matrixPortraitBuilder.Build(grid);
        _preconditionMatrix = globalMatrix.Clone();
        _equation = new Equation<SymmetricSparseMatrix>(
            globalMatrix,
            new Vector(grid.Nodes.Length),
            new Vector(grid.Nodes.Length)
        );

        foreach (var element in grid)
        {
            var localMatrix = _localAssembler.AssembleMatrix(element);
            var localRightPart = _localAssembler.AssembleRightPart(element);

            _inserter.InsertMatrix(_equation.Matrix, localMatrix);
            _inserter.InsertVector(_equation.RightPart, localRightPart);
        }

        return this;
    }

    public GlobalAssembler<TNode> ApplyFirstBoundaries(FirstBoundaryValue[] conditions)
    {
        foreach (var condition in conditions)
        {
            _gaussExсluder.Exclude(_equation, condition);
        }

        return this;
    }

    public Equation<SymmetricSparseMatrix> BuildEquation()
    {
        return _equation;
    }

    public SymmetricSparseMatrix AllocatePreconditionMatrix()
    {
        _preconditionMatrix = _equation.Matrix.Copy(_preconditionMatrix);
        return _preconditionMatrix;
    }
}