using FEM2.Core.Base;
using FEM2.Core.Global;
using FEM2.Core.Local;
using FEM2.FEM.Assembling;

namespace FEM2.TwoDimensional.Assembling;

public class Inserter : IInserter<SymmetricSparseMatrix>
{
    public void InsertMatrix(SymmetricSparseMatrix globalMatrix, LocalMatrix localMatrix)
    {
        var nodesIndexes = localMatrix.Indexes;

        for (var i = 0; i < nodesIndexes.Length; i++)
        {
            var row = nodesIndexes[i];

            for (var j = 0; j < i; j++)
            {
                var column = nodesIndexes[j];

                globalMatrix[row, column] += localMatrix[i, j];
            }

            globalMatrix[row, row] += localMatrix[i, i];
        }
    }

    public void InsertVector(Vector globalVector, LocalVector localVector)
    {
        for (var i = 0; i < localVector.Count; i++)
        {
            globalVector[localVector.Indexes[i]] += localVector[i];
        }
    }
}