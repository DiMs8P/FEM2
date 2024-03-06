using FEM2.Calculus;
using FEM2.Core;
using FEM2.Core.GridComponents;
using FEM2.FEM;
using FEM2.SLAE;
using FEM2.TwoDimensional.Assembling.Local;
using Vector = FEM2.Core.Base.Vector;

namespace FEM2.TwoDimensional;

public class Solver
{
    private readonly Grid<Node2D> _grid;
    private readonly Vector _solution;
    private readonly LocalBasisFunctionsProvider _localBasisFunctionsProvider;
    private readonly DerivativeCalculator _derivativeCalculator;

    public Solver(Grid<Node2D> grid, Vector solution, LocalBasisFunctionsProvider localBasisFunctionsProvider, DerivativeCalculator derivativeCalculator)
    {
        _grid = grid;
        _solution = solution;
        _localBasisFunctionsProvider = localBasisFunctionsProvider;
        _derivativeCalculator = derivativeCalculator;
    }

    public double GetAz(Node2D point)
    {
        if (AreaHas(point))
        {
            var element = FindElementContainingPoint(point);

            var basisFunctions = _localBasisFunctionsProvider.GetBilinearFunctions(element);
            var azValue = CalculateSolutionValue(basisFunctions, element, point);

            CourseHolder.WriteAz(point, azValue);
            return azValue;
        }

        WriteMissedAreaInfo(point);
        return double.NaN;
    }

    public double GetB(Node2D point)
    {
        if (AreaHas(point))
        {
            var element = FindElementContainingPoint(point);

            var basisFunctionsX = _localBasisFunctionsProvider.GetBilinearFunctionsDerivatives(element, 'x');
            var basisFunctionsY = _localBasisFunctionsProvider.GetBilinearFunctionsDerivatives(element, 'y');
            
            var (bx, by) = CalculateBComponents(basisFunctionsX, basisFunctionsY, element, point);
            var bValue = Math.Sqrt(bx * bx + by * by);

            CourseHolder.WriteB(point, bx, by, bValue);
            return bValue;
        }

        WriteMissedAreaInfo(point);
        return double.NaN;
    }

    private Element FindElementContainingPoint(Node2D point)
    {
        return _grid.Elements.First(element => ElementHas(element, point));
    }

    private double CalculateSolutionValue(LocalBasisFunction[] basisFunctions, Element element, Node2D point)
    {
        return element.NodesIndexes
            .Select((nodeIndex, i) => _solution[nodeIndex] * basisFunctions.ElementAt(i).Calculate(point))
            .Sum();
    }

    private (double bx, double by) CalculateBComponents(LocalBasisFunction[] basisFunctionsX, LocalBasisFunction[] basisFunctionsY, Element element, Node2D point)
    {
        var bx = element.NodesIndexes
            .Select((nodeIndex, i) =>  _solution[nodeIndex] * basisFunctionsY.ElementAt(i).Calculate(point))
            .Sum();

        var by = -element.NodesIndexes
            .Select((nodeIndex, i) => _solution[nodeIndex] * basisFunctionsX.ElementAt(i).Calculate(point))
            .Sum();

        return (bx, by);
    }

    private void WriteMissedAreaInfo(Node2D point)
    {
        CourseHolder.WriteAreaInfo();
        CourseHolder.WriteAz(point, double.NaN);
        CourseHolder.WriteB(point, double.NaN, double.NaN, double.NaN);
    }

    private bool ElementHas(Element element, Node2D node) => IsPointWithinBounds(node, _grid.Nodes[element.NodesIndexes[0]], _grid.Nodes[element.NodesIndexes[^1]]);
    private bool AreaHas(Node2D node) => IsPointWithinBounds(node, _grid.Nodes[0], _grid.Nodes[^1]);

    private bool IsPointWithinBounds(Node2D point, Node2D lowerLeft, Node2D upperRight)
    {
        return point.X >= lowerLeft.X && point.X <= upperRight.X && point.Y >= lowerLeft.Y && point.Y <= upperRight.Y;
    }
}