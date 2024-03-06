using FEM2.Core.GridComponents;
using FEM2.TwoDimensional.Assembling.Local;

namespace FEM2.Calculus;

public class DerivativeCalculator
{
    private const double Delta = 1.0e-3;

    public double Calculate(LocalBasisFunction localBasisFunction, Node2D point, char variableChar)
    {
        double result;

        if (variableChar == 'x')
        {
            result = localBasisFunction.Calculate(point.X + Delta, point.Y) - localBasisFunction.Calculate(point.X - Delta, point.Y);
        }
        else
        {
            result = localBasisFunction.Calculate(point.X, point.Y + Delta) - localBasisFunction.Calculate(point.X, point.Y - Delta);
        }
        return result / (2.0 * Delta);
    }
}