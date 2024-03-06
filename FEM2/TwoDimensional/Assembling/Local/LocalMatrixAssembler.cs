using DirectProblem.TwoDimensional.Assembling.MatrixTemplates;
using FEM2.Core.Base;
using FEM2.Core.GridComponents;
using FEM2.FEM.Assembling.Local;
using FEM2.TwoDimensional.Assembling.MatrixTemplates;

namespace FEM2.TwoDimensional.Assembling.Local;

public class LocalMatrixAssembler : ILocalMatrixAssembler
{
    private readonly Matrix _stiffnessTemplate;
    private readonly Matrix _massTemplate;
    private readonly Matrix _stiffness = new(4);
    private readonly Matrix _stiffnessX = new(2);
    private readonly Matrix _stiffnessY = new(2);
    private readonly Matrix _mass = new(4);
    private readonly Matrix _massX = new(2);
    private readonly Matrix _massY = new(2);

    public LocalMatrixAssembler()
    {
        _stiffnessTemplate = StiffnessMatrixTemplateProvider.StiffnessMatrix;
        _massTemplate = MassMatrixTemplateProvider.MassMatrix;
    }

    public Matrix AssembleStiffnessMatrix(Element element)
    {
        var stiffnessX = AssembleStiffnessMatrix(_stiffnessX, element.Length);
        var stiffnessY = AssembleStiffnessMatrix(_stiffnessY, element.Height);

        var massX = AssembleMassMatrix(_massX, element.Length);
        var massY = AssembleMassMatrix(_massY, element.Height);

        for (var i = 0; i < element.NodesIndexes.Length; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                _stiffness[i, j] = stiffnessX[GetMuIndex(i), GetMuIndex(j)] * massY[GetNuIndex(i), GetNuIndex(j)] +
                                   massX[GetMuIndex(i), GetMuIndex(j)] * stiffnessY[GetNuIndex(i), GetNuIndex(j)];
                _stiffness[j, i] = _stiffness[i, j];
            }
        }

        return _stiffness;
    }

    public Matrix AssembleMassMatrix(Element element)
    {
        var massX = AssembleMassMatrix(_massX, element.Length);
        var massY = AssembleMassMatrix(_massY, element.Height);

        for (var i = 0; i < element.NodesIndexes.Length; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                _mass[i, j] = massX[GetMuIndex(i), GetMuIndex(j)] * massY[GetNuIndex(i), GetNuIndex(j)];
                _mass[j, i] = _mass[i, j];
            }
        }

        return _mass;
    }

    private Matrix AssembleStiffnessMatrix(Matrix stiffness, double dimension)
    {
        Matrix.Multiply(1 / dimension, _stiffnessTemplate, stiffness);

        return stiffness;
    }

    private Matrix AssembleMassMatrix(Matrix mass, double dimension)
    {
        Matrix.Multiply(dimension / 6, _massTemplate, mass);

        return mass;
    }

    private int GetMuIndex(int i)
    {
        return i % 2;
    }

    private int GetNuIndex(int i)
    {
        return i / 2;
    }
}