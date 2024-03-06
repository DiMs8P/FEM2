using FEM2.Core.Base;
using FEM2.Core.Global;

namespace FEM2.SLAE.Solvers;

public class LLTSparse
{
    public Vector Solve(SymmetricSparseMatrix matrix, Vector vector, Vector? result = null)
    {
        var y = CalcY(matrix, vector, result);
        var x = CalcX(matrix, y, y);

        return x;
    }

    private Vector CalcY(SymmetricSparseMatrix sparseMatrix, Vector b, Vector? y = null)
    {
        y ??= new Vector(b.Count);

        for (var i = 0; i < sparseMatrix.Count; i++)
        {
            var sum = 0d;

            foreach (var j in sparseMatrix[i])
            {
                sum += sparseMatrix[i, j] * y[j];
            }

            y[i] = (b[i] - sum) / sparseMatrix[i, i];
        }

        return y;
    }

    private Vector CalcX(SymmetricSparseMatrix sparseMatrix, Vector y, Vector? x = null)
    {
        x = x == null ? y.Clone() : y.Copy(x);

        for (var i = sparseMatrix.Count - 1; i >= 0; i--)
        {
            x[i] /= sparseMatrix[i, i];

            var columns = sparseMatrix[i];
            for (var j = columns.Length - 1; j >= 0; j--)
            {
                x[columns[j]] -= sparseMatrix[columns[j], i] * x[i];
            }
        }

        return x;
    }
}