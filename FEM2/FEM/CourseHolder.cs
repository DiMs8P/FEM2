using FEM2.Core.GridComponents;

namespace FEM2.FEM;

public class CourseHolder
{
    public static void GetInfo(int iteration, double residual)
    {
        Console.Write($"Iteration: {iteration}, residual: {residual:E14}                                   \r");
    }

    public static void WriteAz(Node2D point, double value)
    {
        Console.WriteLine($"{value:E14}");
    }

    public static void WriteB(Node2D point, double bX, double bY, double b)
    {
        Console.WriteLine($"{bX:E14} {bY:E14} {b:E14}");
    }

    public static void WriteAreaInfo()
    {
        Console.WriteLine("Point not in area or time interval");
    }
}