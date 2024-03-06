using FEM2.Core.Local;

namespace FEM2.Core.Boundary;

public record struct FirstBoundary(int NodeIndex);
public record struct FirstBoundaryValue(int NodeIndex, double Value);