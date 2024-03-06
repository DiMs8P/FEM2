using FEM2.Core.GridComponents;
using FEM2.Core.Local;

namespace FEM2.FEM.Assembling.Local;

public interface ILocalAssembler
{
    public LocalMatrix AssembleMatrix(Element element);
    public LocalVector AssembleRightPart(Element element);
}