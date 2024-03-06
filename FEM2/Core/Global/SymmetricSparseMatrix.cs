using FEM2.Core.Base;

namespace FEM2.Core.Global;

public class SymmetricSparseMatrix
{
    private readonly double[] _diagonal;
    private readonly double[] _values;
    private readonly int[] _rowsIndexes;
    private readonly int[] _columnsIndexes;
    public ReadOnlySpan<int> RowsIndexes => new(_rowsIndexes);
    public ReadOnlySpan<int> ColumnsIndexes => new(_columnsIndexes);

    public int Count => _diagonal.Length;

    public ReadOnlySpan<int> this[int rowIndex] => ColumnsIndexes[RowsIndexes[rowIndex]..RowsIndexes[rowIndex + 1]];
    public double this[int rowIndex, int columnIndex]
    {
        get
        {
            if (rowIndex < 0 || columnIndex < 0) throw new ArgumentOutOfRangeException(nameof(rowIndex));

            if (rowIndex == columnIndex)
            {
                return _diagonal[rowIndex];
            }

            if (columnIndex > rowIndex) (rowIndex, columnIndex) = (columnIndex, rowIndex);

            var index = IndexOf(rowIndex, columnIndex);
            return _values[index];
        }
        set
        {
            if (rowIndex < 0 || columnIndex < 0) throw new ArgumentOutOfRangeException(nameof(rowIndex));

            if (rowIndex == columnIndex)
            {
                _diagonal[rowIndex] = value;
                return;
            }

            if (columnIndex > rowIndex) (rowIndex, columnIndex) = (columnIndex, rowIndex);

            var index = IndexOf(rowIndex, columnIndex);
            _values[index] = value;
        }
    }

    public int IndexOf(int rowIndex, int columnIndex) => Array.IndexOf(_columnsIndexes, columnIndex, RowsIndexes[rowIndex],
        RowsIndexes[rowIndex + 1] - RowsIndexes[rowIndex]);

    public SymmetricSparseMatrix(int[] rowsIndexes, int[] columnsIndexes)
    {
        _rowsIndexes = rowsIndexes;
        _columnsIndexes = columnsIndexes;
        _diagonal = new double[rowsIndexes.Length - 1];
        _values = new double[rowsIndexes[^1]];
    }

    public SymmetricSparseMatrix
    (
        int[] rowsIndexes,
        int[] columnsIndexes,
        double[] diagonal,
        double[] values
    )
    {
        _rowsIndexes = rowsIndexes;
        _columnsIndexes = columnsIndexes;
        _diagonal = diagonal;
        _values = values;
    }

    public static Vector Multiply(SymmetricSparseMatrix matrix, Vector vector, Vector? result = null)
    {
        if (matrix.Count != vector.Count)
            throw new ArgumentOutOfRangeException(
                $"{nameof(matrix)} and {nameof(vector)} must have same size");

        if (result == null) result = new Vector(matrix.Count);
        else result.Clear();

        for (var i = 0; i < matrix.Count; i++)
        {
            result[i] += matrix[i, i] * vector[i];

            foreach (var j in matrix[i])
            {
                result[i] += matrix[i, j] * vector[j];
                result[j] += matrix[j, i] * vector[i];
            }
        }

        return result;
    }

    public static SymmetricSparseMatrix Multiply(double number, SymmetricSparseMatrix matrix, SymmetricSparseMatrix? result = null)
    {
        result ??= matrix.Clone();

        for (var i = 0; i < matrix.Count; i++)
        {
            result[i, i] = number * matrix[i, i];

            foreach (var j in matrix[i])
            {
                result[i, j] = number * matrix[i, j];
            }
        }

        return result;
    }

    public static SymmetricSparseMatrix Sum(SymmetricSparseMatrix matrix1, SymmetricSparseMatrix matrix2, SymmetricSparseMatrix? result = null)
    {
        result ??= matrix1.Clone();

        for (var i = 0; i < matrix1.Count; i++)
        {
            result[i, i] = matrix1[i, i] + matrix2[i, i];

            foreach (var j in matrix1[i])
            {
                result[i, j] = matrix1[i, j] + matrix2[i, j];
            }
        }

        return matrix1;
    }

    public SymmetricSparseMatrix Clone()
    {
        var rowIndexes = new int[_rowsIndexes.Length];
        var columnIndexes = new int[_columnsIndexes.Length];
        var diagonal = new double[_diagonal.Length];
        var values = new double[_values.Length];

        Array.Copy(_rowsIndexes, rowIndexes, _rowsIndexes.Length);
        Array.Copy(_columnsIndexes, columnIndexes, _columnsIndexes.Length);
        Array.Copy(_diagonal, diagonal, _diagonal.Length);
        Array.Copy(_values, values, _values.Length);

        return new SymmetricSparseMatrix(rowIndexes, columnIndexes, diagonal, values);
    }

    public SymmetricSparseMatrix Copy(SymmetricSparseMatrix sparseMatrix)
    {
        Array.Copy(_diagonal, sparseMatrix._diagonal, _diagonal.Length);
        Array.Copy(_values, sparseMatrix._values, _values.Length);

        return sparseMatrix;
    }
}