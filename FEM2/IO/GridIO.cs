using System.Text;
using FEM2.Core.Boundary;
using FEM2.Core.GridComponents;

namespace FEM2.IO
{
    public class GridIO
    {
        public GridIO()
        {
        }

        public Node2D[] ReadNodes(string fileName)
        {
            using var binaryReader = new BinaryReader(File.Open(fileName, FileMode.Open), Encoding.ASCII);

            var nodes = new List<Node2D>();

            while (binaryReader.PeekChar() > -1)
            {
                var x = binaryReader.ReadDouble();
                var y = binaryReader.ReadDouble();

                nodes.Add(new Node2D(x, y));
            }

            return nodes.ToArray();
        }

        public List<int> ReadMaterials(string fileName)
        {
            using var binaryReader = new BinaryReader(File.Open(fileName, FileMode.Open), Encoding.ASCII);

            var materials = new List<int>();

            while (binaryReader.PeekChar() > -1)
            {
                materials.Add(binaryReader.ReadInt32() - 1);
            }

            return materials;
        }

        public Element[] ReadElements(Node2D[] nodes, List<int> materials, string fileName)
        {
            using var binaryReader = new BinaryReader(File.Open(fileName, FileMode.Open), Encoding.ASCII);

            var elements = new List<Element>();

            var i = 0;

            while (binaryReader.PeekChar() > -1)
            {
                var elementNodes = new int[4];

                elementNodes[2] = binaryReader.ReadInt32() - 1;
                elementNodes[3] = binaryReader.ReadInt32() - 1;
                elementNodes[0] = binaryReader.ReadInt32() - 1;
                elementNodes[1] = binaryReader.ReadInt32() - 1;

                binaryReader.ReadInt32();
                binaryReader.ReadInt32();

                var length = nodes[elementNodes[1]].X - nodes[elementNodes[0]].X;
                var height = nodes[elementNodes[2]].Y - nodes[elementNodes[0]].Y;

                elements.Add(new Element(elementNodes, length, height, materials[i]));

                i++;
            }

            return elements.ToArray();
        }

        public FirstBoundary[] ReadFirstBoundaries(string fileName)
        {
            using var binaryReader = new BinaryReader(File.Open(fileName, FileMode.Open), Encoding.ASCII);

            var boundaries = new List<FirstBoundary>();

            while (binaryReader.PeekChar() > -1)
            {
                boundaries.Add(new FirstBoundary(binaryReader.ReadInt32() - 1));
            }

            return boundaries.ToArray();
        }
    }
}
