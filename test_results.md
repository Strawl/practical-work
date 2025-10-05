# Solver Timing Comparison (3D Linear Elasticity)

# Solver Timing Comparison (3D Linear Elasticity)

| Framework / Mode   | Solver                        | Time (s)   |
|--------------------|-------------------------------|------------|
| **JAX-FEM**        | BiCGStab                      | 1.090      |
|                    | GMRES + Jacobi (PETSc)        | 1.030      |
|                    | BiCGStab(L) + ILU (PETSc)     | 0.717      |
|                    | UMFPACK (direct solver)       | 0.857      |
| **FeAX (JIT)**     | BiCGStab                      | 0.695      |
|                    | GMRES + Jacobi preconditioner | 0.704      |
|                    | CG (benchmark)                | 0.571      |
| **FeAX (no JIT)**  | BiCGStab                      | 2.134      |
|                    | GMRES + Jacobi preconditioner | 2.230      |
|                    | CG (benchmark)                | 2.067      |