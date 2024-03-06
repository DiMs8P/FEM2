using FEM2.Core.Base;

namespace DirectProblem.TwoDimensional.Assembling.MatrixTemplates;

public class MassMatrixTemplateProvider
{
    public static Matrix MassMatrix => new(
        new[,]
        {
            { 2d, 1d },
            { 1d, 2d }
        });
}