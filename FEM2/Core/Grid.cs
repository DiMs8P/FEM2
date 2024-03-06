using FEM2.Core.GridComponents;

namespace FEM2.Core;

public class Grid<TPoint>
{
    public TPoint[] Nodes { get; }
    public Element[] Elements { get; }

    public IEnumerator<Element> GetEnumerator() => ((IEnumerable<Element>)Elements).GetEnumerator();

    public Grid(TPoint[] nodes, Element[] elements)
    {
        Nodes = nodes;
        Elements = elements;
    }
}