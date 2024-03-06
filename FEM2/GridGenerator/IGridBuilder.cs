using FEM2.Core;

namespace DirectProblem.GridGenerator;

public interface IGridBuilder<TPoint>
{
    public Grid<TPoint> Build();
}