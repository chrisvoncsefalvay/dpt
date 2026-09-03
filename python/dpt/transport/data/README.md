# Sobol direction numbers

`sobol_joe_kuo_6_21201.npz` contains the primitive polynomials and initial
directions for Stephen Joe and Frances Kuo's D(6) construction, covering 21,201
dimensions. The authors recommend this set, updated on 16 September 2010, at
<https://web.maths.unsw.edu.au/~fkuo/sobol/>. The original data are
<https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201>.

The archive is reproduced byte for byte from SciPy 1.18.1's
`scipy/stats/_sobol_direction_numbers.npz`. Its SHA-256 is
`4859931147d42ce465b8605cb277f957d98b839d03194fdf06579357906d193b`.
SciPy documents the archive conversion in
<https://github.com/scipy/scipy/blob/v1.16.0/scipy/stats/_sobol.pyx>.
The data arrays are `poly`, with shape `(21201,)`, and `vinit`, with shape
`(21201, 18)`. DPT constructs the binary directions and randomisations itself.
SciPy is not required at runtime.

The original direction numbers retain the complete authors' BSD notice in
`JOE_KUO_LICENSE.txt`, obtained from
<https://web.maths.unsw.edu.au/~fkuo/sobol/licence>. The SciPy archive packaging
retains its complete BSD notice in `SCIPY_LICENSE.txt`. These licences apply
to this attributed data asset and its packaging. DPT's own code remains
Apache-2.0.

Reference: S. Joe and F. Y. Kuo, “Constructing Sobol sequences with better
two-dimensional projections”, SIAM Journal on Scientific Computing 30(5)
(2008), 2635–2654, <https://doi.org/10.1137/070709359>.
