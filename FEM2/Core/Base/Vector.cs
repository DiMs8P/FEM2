namespace FEM2.Core.Base;

public class Vector
{
    public double[] Values { get; }

    public Vector() : this(Array.Empty<double>()) { }
    public Vector(double[] values)
    {
        Values = values;
    }
    public Vector(int size) : this(new double[size]) { }

    public double this[int index]
    {
        get => Values[index];
        set => Values[index] = value;
    }

    public int Count => Values.Length;
    public double Norm => Math.Sqrt(ScalarProduct(this, this));

    public static double ScalarProduct(Vector vector1, Vector vector2)
    {
        var result = 0d;
        for (var i = 0; i < vector1.Count; i++)
        {
            result += vector1[i] * vector2[i];
        }
        return result;
    }

    public double ScalarProduct(Vector vector)
    {
        return ScalarProduct(this, vector);
    }

    public static Vector Sum(Vector vector1, Vector vector2, Vector? result = null)
    {
        if (vector1.Count != vector2.Count)
            throw new ArgumentOutOfRangeException(
                $"{nameof(vector1)} and {nameof(vector2)} must have same size");

        result ??= new Vector(vector1.Values);

        for (var i = 0; i < vector1.Count; i++)
        {
            result[i] = vector1[i] + vector2[i];
        }

        return result;
    }

    public static Vector Subtract(Vector vector1, Vector vector2, Vector? result = null)
    {
        result ??= new Vector(vector1.Count);

        if (vector1.Count != vector2.Count)
            throw new ArgumentOutOfRangeException(
                $"{nameof(vector1)} and {nameof(vector2)} must have same size");

        for (var i = 0; i < vector1.Count; i++)
        {
            result[i] = vector1[i] - vector2[i];
        }

        return result;
    }

    public static Vector Multiply(double number, Vector localVector, Vector? result = null)
    {
        result ??= new Vector(localVector.Values);

        for (var i = 0; i < localVector.Count; i++)
        {
            result[i] = number * localVector[i];
        }

        return result;
    }

    public void Fill(double value)
    {
        Array.Fill(Values, value);
    }

    public void Clear()
    {
        Array.Clear(Values);
    }

    public Vector Clone()
    {
        var clone = (double[])Values.Clone();

        return new Vector(clone);
    }

    public Vector Copy(Vector vector)
    {
        if (Count != vector.Count)
            throw new ArgumentOutOfRangeException(
                $"{nameof(Values)}  and  {nameof(vector)} must have same size");

        Array.Copy(Values, vector.Values, Count);

        return vector;
    }

    public IEnumerator<double> GetEnumerator() => ((IEnumerable<double>)Values).GetEnumerator();
}