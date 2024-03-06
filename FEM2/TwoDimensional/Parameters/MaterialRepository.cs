using FEM2.Core.GridComponents;

namespace FEM2.TwoDimensional.Parameters;

public class MaterialRepository
{
    private readonly Dictionary<int, Material> _materials;

    public MaterialRepository(List<double> mus, List<double> js)
    {
        _materials = new Dictionary<int, Material>(mus.Count);

        for (var i = 0; i < mus.Count; i++)
        {
            _materials.Add(i, new Material(mus[i], js[i]));
        }
    }

    public Material GetById(int id)
    {
        return _materials[id];
    }
}