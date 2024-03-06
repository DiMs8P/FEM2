using FEM2.Core;
using FEM2.Core.Global;
using FEM2.Core.GridComponents;
using FEM2.FEM.Assembling;

namespace FEM2.TwoDimensional.Assembling;

public class MatrixPortraitBuilder : IMatrixPortraitBuilder<Node2D, SymmetricSparseMatrix>
{
    private List<SortedSet<int>> _adjacencyList = null!;

    public SymmetricSparseMatrix Build(Grid<Node2D> grid)
    {
        BuildAdjacencyList(grid);

        var amount = 0;
        var buf = _adjacencyList.Select(nodeSet => amount += nodeSet.Count).ToList();
        buf.Insert(0, 0);

        var rowsIndexes = buf.ToArray();
        var columnsIndexes = _adjacencyList.SelectMany(nodeList => nodeList).ToArray();

        return new SymmetricSparseMatrix(rowsIndexes, columnsIndexes);
    }

    private void BuildAdjacencyList(Grid<Node2D> grid)
    {
        _adjacencyList = new List<SortedSet<int>>(grid.Nodes.Length);

        for (var i = 0; i < grid.Nodes.Length; i++)
        {
            _adjacencyList.Add([]);
        }

        foreach (var element in grid)
        {
            var nodesIndexes = element.NodesIndexes;

            foreach (var currentNode in nodesIndexes)
            {
                foreach (var nodeIndex in nodesIndexes)
                {
                    if (currentNode > nodeIndex) _adjacencyList[currentNode].Add(nodeIndex);
                }
            }
        }
    }
}