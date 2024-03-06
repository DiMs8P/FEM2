using FEM2.Core;
using FEM2.Core.Base;
using FEM2.Core.GridComponents;
using FEM2.Core.Local;
using FEM2.FEM.Assembling.Local;
using FEM2.TwoDimensional.Parameters;

namespace FEM2.TwoDimensional.Assembling.Local;

public class LocalAssembler : ILocalAssembler
{
    private readonly Grid<Node2D> _grid;
    private readonly ILocalMatrixAssembler _localMatrixAssembler;
    private readonly MaterialRepository _materialRepository;
    private readonly Vector _rightPart = new(4);
    private readonly Vector _buffer = new(4);

    public LocalAssembler
    (
        Grid<Node2D> grid,
        ILocalMatrixAssembler localMatrixAssembler,
        MaterialRepository materialRepository
    )
    {
        _grid = grid;
        _localMatrixAssembler = localMatrixAssembler;
        _materialRepository = materialRepository;
    }

    public LocalMatrix AssembleMatrix(Element element)
    {
        var matrix = GetStiffnessMatrix(element);

        return new LocalMatrix(element.NodesIndexes, matrix);
    }

    public LocalVector AssembleRightPart(Element element)
    {
        var vector = GetRightPart(element);

        return new LocalVector(element.NodesIndexes, vector);
    }

    private Matrix GetStiffnessMatrix(Element element)
    {
        var mu = _materialRepository.GetById(element.MaterialId).Mu;
        var stiffness = _localMatrixAssembler.AssembleStiffnessMatrix(element);

        return Matrix.Multiply(1d / mu, stiffness, stiffness);
    }

    private Vector GetRightPart(Element element)
    {
        var j = _materialRepository.GetById(element.MaterialId).J;
        var mass = _localMatrixAssembler.AssembleMassMatrix(element);

        _rightPart.Fill(j);

        return Matrix.Multiply(mass, _rightPart, _buffer);
    }
}