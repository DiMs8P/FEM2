namespace FEM2.Core.Base;

public class Matrix
{
    public double[,] Values { get; }

    public Matrix(double[,] matrix)
    {
        Values = matrix;
    }
    public Matrix(int n) : this(new double[n, n]) { }

    public int CountRows => Values.GetLength(0);
    public int CountColumns => Values.GetLength(1);

    public double this[int i, int j]
    {
        get => Values[i, j];
        set => Values[i, j] = value;
    }

    public static Matrix Sum(Matrix matrix1, Matrix matrix2, Matrix? result = null)
    {
        if (matrix1.CountRows != matrix2.CountRows || matrix1.CountColumns != matrix2.CountColumns)
            throw new ArgumentOutOfRangeException(
                $"{nameof(matrix1)} and {nameof(matrix2)} must have same size");

        result ??= new Matrix(matrix1.CountRows);

        for (var i = 0; i < matrix1.CountRows; i++)
        {
            for (var j = 0; j < matrix1.CountColumns; j++)
            {
                result[i, j] = matrix1[i, j] + matrix2[i, j];
            }
        }

        return result;
    }

    public static Matrix Multiply(double coefficient, Matrix matrix, Matrix? result = null)
    {
        result ??= new Matrix(matrix.CountRows);

        for (var i = 0; i < matrix.CountRows; i++)
        {
            for (var j = 0; j < matrix.CountColumns; j++)
            {
                result[i, j] = matrix[i, j] * coefficient;
            }
        }

        return result;
    }

    public static Vector Multiply(Matrix matrix, Vector vector, Vector? result = null)
    {
        if (matrix.CountRows != vector.Count)
            throw new ArgumentOutOfRangeException(
                $"{nameof(matrix)} and {nameof(vector)} must have same size");

        if (result is null) result = new Vector(vector.Count);
        else result.Clear();

        for (var i = 0; i < matrix.CountRows; i++)
        {
            for (var j = 0; j < matrix.CountColumns; j++)
            {
                result[i] += matrix[i, j] * vector[j];
            }
        }

        return result;
    }
}