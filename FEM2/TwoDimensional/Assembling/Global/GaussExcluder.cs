using FEM2.Core.Boundary;
using FEM2.Core.Global;
using FEM2.FEM.Assembling.Global;

namespace FEM2.TwoDimensional.Assembling.Global;

public class GaussExcluder : IGaussExcluder<SymmetricSparseMatrix>
{
    public void Exclude(Equation<SymmetricSparseMatrix> equation, FirstBoundaryValue boundary)
    {
        var row = boundary.NodeIndex;
        equation.RightPart[row] = boundary.Value;
        equation.Matrix[row, row] = 1d;

        foreach (var j in equation.Matrix[row])
        {
            equation.RightPart[j] -= equation.Matrix[row, j] * boundary.Value;
            equation.Matrix[row, j] = 0d;
        }

        for (var column = row + 1; column < equation.Matrix.Count; column++)
        {
            if (!equation.Matrix[column].Contains(row)) continue;

            equation.RightPart[column] -= equation.Matrix[row, column] * boundary.Value;
            equation.Matrix[row, column] = 0d;
        }
    }
}