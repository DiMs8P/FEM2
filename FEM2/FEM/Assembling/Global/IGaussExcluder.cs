using FEM2.Core.Boundary;
using FEM2.Core.Global;

namespace FEM2.FEM.Assembling.Global;

public interface IGaussExcluder<TMatrix>
{
    public void Exclude(Equation<TMatrix> equation, FirstBoundaryValue boundaryValue);
}