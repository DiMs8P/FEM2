namespace DirectProblem.SLAE.Preconditions;

public interface IPreconditioner<TMatrix>
{
    public TMatrix Decompose(TMatrix globalMatrix);
}