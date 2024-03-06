using FEM2.Core.Base;

namespace FEM2.TwoDimensional.Assembling.MatrixTemplates;

public class StiffnessMatrixTemplateProvider
{
    public static Matrix StiffnessMatrix => new(
        new[,]
        {
            { 1d, -1d },
            { -1d, 1d }
        });
}