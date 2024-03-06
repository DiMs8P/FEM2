using DirectProblem.GridGenerator;
using FEM2.Core;
using FEM2.Core.GridComponents;

namespace FEM2.GridGenerator;

public class GridBuilder2D : IGridBuilder<Node2D>
{
    private Node2D[]? _nodes;
    private Element[]? _elements;

    public GridBuilder2D SetNodes(Node2D[] nodes)
    {
        _nodes = nodes;
        return this;
    }

    public GridBuilder2D SetElements(Element[] elements)
    {
        _elements = elements;
        return this;
    }

    public Grid<Node2D> Build()
    {
        return new Grid<Node2D>(_nodes, _elements);
    }
}