using FEM2.Core.Base;

namespace FEM2.Core.Global;

public record Equation<TMatrix>(TMatrix Matrix, Vector Solution, Vector RightPart);