using FEM2.Core;

namespace FEM2.FEM.Assembling;

public interface IMatrixPortraitBuilder<TNode, out TMatrix>
{
    TMatrix Build(Grid<TNode> grid);
}