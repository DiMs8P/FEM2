using FEM2.Core;
using FEM2.Core.Base;
using FEM2.Core.Boundary;
using FEM2.Core.GridComponents;
using FEM2.Core.Local;

namespace FEM2.TwoDimensional.Assembling.Boundary;

public class FirstBoundaryProvider
{
    private readonly Grid<Node2D> _grid;
    private FirstBoundaryValue[]? _conditionsValues;

    public FirstBoundaryProvider(Grid<Node2D> grid)
    {
        _grid = grid;
    }

    public FirstBoundaryValue[] GetConditions(FirstBoundary[] conditions)
    {
        ValidateMemory(conditions);

        for (var i = 0; i < _conditionsValues.Length; i++)
        {
            _conditionsValues[i].NodeIndex = conditions[i].NodeIndex;
            _conditionsValues[i].Value = 0d;
        }

        return _conditionsValues;
    }

    public FirstBoundary[] GetArrays(int elementsByLength, int elementsByHeight)
    {
        var conditions = new FirstBoundary[elementsByLength + 2 * elementsByHeight];

        //var j = 0;

        //for (var i = 0; i < elementsByHeight; i++, j++)
        //{
        //    conditions[j] = new FirstCondition(i * elementsByLength, Bound.Left);
        //}

        //for (var i = 0; i < elementsByHeight; i++, j++)
        //{
        //    conditions[j] = new FirstCondition((i + 1) * elementsByLength - 1, Bound.Right);
        //}

        //for (var i = elementsByLength * (elementsByHeight - 1); i < elementsByLength * elementsByHeight; i++, j++)
        //{
        //    conditions[j] = new FirstCondition(i, Bound.Upper);
        //}

        return conditions;
    }

    private void ValidateMemory(FirstBoundary[] conditions)
    {
        if (_conditionsValues is null || _conditionsValues.Length != conditions.Length)
        {
            _conditionsValues = new FirstBoundaryValue[conditions.Length];
        }
    }
}