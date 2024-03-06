using FEM2.Core.Base;
using FEM2.Core.GridComponents;

namespace FEM2.FEM.Assembling.Local;

public interface ILocalMatrixAssembler
{
    public Matrix AssembleStiffnessMatrix(Element element);
    public Matrix AssembleMassMatrix(Element element);
}