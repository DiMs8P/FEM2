using FEM2.Core.Boundary;

namespace FEM2.Core.GridComponents;

public class Element
{
    public int[] NodesIndexes { get; }
    public int MaterialId { get; }
    public double Length { get; }
    public double Height { get; }

    public IEnumerator<int> GetEnumerator() => ((IEnumerable<int>)NodesIndexes).GetEnumerator();

    public Element(int[] nodesIndexes, double length, double height, int materialId)
    {
        NodesIndexes = nodesIndexes;
        Length = length;
        Height = height;
        MaterialId = materialId;
    }

    public (int[], double) GetBoundNodeIndexes(Bound bound, int[]? indexes = null)
    {
        indexes ??= new int[2];

        switch (bound)
        {
            case Bound.Lower:
                indexes[0] = NodesIndexes[0];
                indexes[1] = NodesIndexes[1];
                return (indexes, Length);
            case Bound.Left:
                indexes[0] = NodesIndexes[0];
                indexes[1] = NodesIndexes[2];
                return (indexes, Height);
            case Bound.Right:
                indexes[0] = NodesIndexes[1];
                indexes[1] = NodesIndexes[3];
                return (indexes, Height);
            case Bound.Upper:
                indexes[0] = NodesIndexes[2];
                indexes[1] = NodesIndexes[3];
                return (indexes, Length);
            default:
                throw new ArgumentOutOfRangeException();
        }
    }
}