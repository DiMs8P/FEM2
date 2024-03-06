using FEM2.Core;
using FEM2.Core.GridComponents;
using FEM2.FEM;

namespace FEM2.TwoDimensional.Assembling.Local;

public class LocalBasisFunctionsProvider
{
    private readonly Grid<Node2D> _grid;
    private readonly LinearFunctionsProvider _linearFunctionsProvider;
    private readonly LocalBasisFunction[] _localBasisFunctions = new LocalBasisFunction[4];
    private readonly LocalBasisFunction[] _localBasisFunctionsDerivativesByX = new LocalBasisFunction[4];
    private readonly LocalBasisFunction[] _localBasisFunctionsDerivativesByY = new LocalBasisFunction[4];

    public LocalBasisFunctionsProvider(Grid<Node2D> grid, LinearFunctionsProvider linearFunctionsProvider)
    {
        _grid = grid;
        _linearFunctionsProvider = linearFunctionsProvider;
    }

    public LocalBasisFunction[] GetBilinearFunctions(Element element)
    {
        var firstXFunction =
            _linearFunctionsProvider.CreateFirstFunction(_grid.Nodes[element.NodesIndexes[1]].X, element.Length);
        var secondXFunction =
            _linearFunctionsProvider.CreateSecondFunction(_grid.Nodes[element.NodesIndexes[0]].X, element.Length);
        var firstYFunction =
            _linearFunctionsProvider.CreateFirstFunction(_grid.Nodes[element.NodesIndexes[2]].Y, element.Height);
        var secondYFunction =
            _linearFunctionsProvider.CreateSecondFunction(_grid.Nodes[element.NodesIndexes[0]].Y, element.Height);

        _localBasisFunctions[0] = new LocalBasisFunction(firstXFunction, firstYFunction);
        _localBasisFunctions[1] = new LocalBasisFunction(secondXFunction, firstYFunction);
        _localBasisFunctions[2] = new LocalBasisFunction(firstXFunction, secondYFunction);
        _localBasisFunctions[3] = new LocalBasisFunction(secondXFunction, secondYFunction);

        return _localBasisFunctions;
    }

    public LocalBasisFunction[] GetBilinearFunctionsDerivatives(Element element, char variable)
    {
        var firstXFunction =
            _linearFunctionsProvider.CreateFirstFunction(_grid.Nodes[element.NodesIndexes[1]].X, element.Length);
        var secondXFunction =
            _linearFunctionsProvider.CreateSecondFunction(_grid.Nodes[element.NodesIndexes[0]].X, element.Length);
        var firstYFunction =
            _linearFunctionsProvider.CreateFirstFunction(_grid.Nodes[element.NodesIndexes[2]].Y, element.Height);
        var secondYFunction =
            _linearFunctionsProvider.CreateSecondFunction(_grid.Nodes[element.NodesIndexes[0]].Y, element.Height);

        if (variable == 'x')
        {
            _localBasisFunctionsDerivativesByX[0] = new LocalBasisFunction(_ => -1 / element.Length, firstYFunction);
            _localBasisFunctionsDerivativesByX[1] = new LocalBasisFunction(_ => 1 / element.Length, firstYFunction);
            _localBasisFunctionsDerivativesByX[2] = new LocalBasisFunction(_ => -1 / element.Length, secondYFunction);
            _localBasisFunctionsDerivativesByX[3] = new LocalBasisFunction(_ => 1 / element.Length, secondYFunction);

            return _localBasisFunctionsDerivativesByX;
        }
        else
        {
            _localBasisFunctionsDerivativesByY[0] = new LocalBasisFunction(firstXFunction, _ => -1 / element.Height);
            _localBasisFunctionsDerivativesByY[1] = new LocalBasisFunction(secondXFunction, _ => -1 / element.Height);
            _localBasisFunctionsDerivativesByY[2] = new LocalBasisFunction(firstXFunction, _ => 1 / element.Height);
            _localBasisFunctionsDerivativesByY[3] = new LocalBasisFunction(secondXFunction, _ => 1 / element.Height);

            return _localBasisFunctionsDerivativesByY;
        }
    }
}