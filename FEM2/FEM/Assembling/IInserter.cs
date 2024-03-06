using FEM2.Core.Base;
using FEM2.Core.Local;

namespace FEM2.FEM.Assembling;

public interface IInserter<in TMatrix>
{
    public void InsertMatrix(TMatrix globalMatrix, LocalMatrix localMatrix);
    public void InsertVector(Vector vector, LocalVector localVector);
}