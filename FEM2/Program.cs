using FEM2.Calculus;
using FEM2.Core.GridComponents;
using FEM2.SLAE.Preconditions;
using FEM2.SLAE.Solvers;
using FEM2.TwoDimensional.Assembling;
using FEM2.TwoDimensional.Assembling.Boundary;
using FEM2.TwoDimensional.Assembling.Global;
using FEM2.TwoDimensional.Assembling.Local;
using FEM2.TwoDimensional.Parameters;
using FEM2.Core;
using FEM2.Core.Boundary;
using FEM2.FEM;
using FEM2.GridGenerator;
using FEM2.IO;
using FEM2.TwoDimensional;

class Program
{
    static void Main()
    {
        var GridIO = new GridIO();
        var Grid = InitGrid(GridIO);
        
        var Materials = InitMaterials();
      
        var (grid, FirstBoundaries) = PrepareGridAndBoundaries(GridIO, Grid);

        var Solution = SolveEquation(grid, FirstBoundaries, Materials);
      
        CalculateAndPrintResults(Solution);
    }

    static GridBuilder2D InitGrid(GridIO griIO)
    {
        var Nodes = griIO.ReadNodes("rz.dat");
        var MaterialsIds = griIO.ReadMaterials("nvkat2d.dat");
        var Elements = griIO.ReadElements(Nodes, MaterialsIds, "nvtr.dat");

        return new GridBuilder2D().SetNodes(Nodes).SetElements(Elements);
    }

    static MaterialRepository InitMaterials()
    {
        const double nu0 = 4 * Math.PI * 1e-7;
        var nuIron = 1000d * nu0;
        var nuAir = 1d * nu0;

        return new MaterialRepository(new List<double>() { nuAir, nuIron, nuAir, nuAir }, new List<double>() { 0d, 0d, 1e7, -1e7 });
    }

    static (Grid<Node2D> grid, FirstBoundary[] firstBoundaries) PrepareGridAndBoundaries(GridIO gridI, GridBuilder2D gridCreator)
    {
        var grid = gridCreator.Build();
        var firstBoundaries = gridI.ReadFirstBoundaries("l1.dat");
        return (grid, firstBoundaries);
    }

    static Solver SolveEquation(Grid<Node2D> grid, FirstBoundary[] firstBoundaries, MaterialRepository materialRepository)
    {
        var localBasisFunctionsProvider = new LocalBasisFunctionsProvider(grid, new LinearFunctionsProvider());
        var localAssembler = new LocalAssembler(grid, new LocalMatrixAssembler(), materialRepository);
        var globalAssembler = new GlobalAssembler<Node2D>(grid, new MatrixPortraitBuilder(), localAssembler, new Inserter(), new GaussExcluder());

        var firstBoundaryProvider = new FirstBoundaryProvider(grid);
        var firstBoundariesValues = firstBoundaryProvider.GetConditions(firstBoundaries);

        var equation = globalAssembler.AssembleEquation(grid).ApplyFirstBoundaries(firstBoundariesValues).BuildEquation();
        var preconditionMatrix = globalAssembler.AllocatePreconditionMatrix();
        var solver = new MCG(new LLTPreconditioner(), new LLTSparse());
        var solution = solver.SetPrecondition(preconditionMatrix).Solve(equation);

        return new Solver(grid, solution, localBasisFunctionsProvider, new DerivativeCalculator());
    }

    static void CalculateAndPrintResults(Solver solver)
    {
        var evaluationPoints = new[]
        {
            new Node2D(-1.4900000E-02, 1.4000000E-03),
            new Node2D(-6.3000000E-03, 3.0000000E-03),
            new Node2D(0.0000000E+00, 2.2000000E-03),
            new Node2D(6.3000000E-03, 3.0000000E-03),
            new Node2D(1.4100000E-02, 3.3000000E-03),
        };

        Console.WriteLine("Results for Az:");
        foreach (var point in evaluationPoints)
        {
            solver.GetAz(point);
        }

        Console.WriteLine("\nResults for B:");
        foreach (var point in evaluationPoints)
        {
            solver.GetB(point);
        }
    }
}
