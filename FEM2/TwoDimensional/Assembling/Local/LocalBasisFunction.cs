using FEM2.Core.GridComponents;

namespace FEM2.TwoDimensional.Assembling.Local;

public class LocalBasisFunction
{
    private readonly Func<double, double> _xFunction;
    private readonly Func<double, double> _yFunction;

    public LocalBasisFunction(Func<double, double> xFunction, Func<double, double> yFunction)
    {
        _xFunction = xFunction;
        _yFunction = yFunction;
    }

    public double Calculate(Node2D node)
    {
        return Calculate(node.X, node.Y);
    }

    public double Calculate(double x, double y)
    {
        return _xFunction(x) * _yFunction(y);
    }
}