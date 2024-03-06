using FEM2.Core.Base;

namespace FEM2.Core.Local;

public class LocalVector
{
    public int[] Indexes { get; }
    public Vector Vector { get; }

    public int Count => Vector.Count;

    public LocalVector(int[] indexes, Vector vector)
    {
        Indexes = indexes;
        Vector = vector;
    }

    public double this[int index]
    {
        get => Vector[index];
        set => Vector[index] = value;
    }

    public IEnumerator<double> GetEnumerator() => Vector.GetEnumerator();
}