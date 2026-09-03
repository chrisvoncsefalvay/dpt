# Material coefficients and reference data

A material coefficient belongs to a particular interaction and composition.
Use total attenuation for primary projection, photoelectric coefficients for a
photoelectric branch, and energy-absorption coefficients for the quantity their
source defines. `PhotonCoefficientTable.kind` keeps that distinction explicit.

## Query xraylib

Install the optional provider with `python -m pip install '.[physical-data]'`.
Supply an atomic number or chemical composition, energy nodes in keV and a
reference density in g/cm³:

```python
from dpt.physical_data import xraylib_coefficients

aluminum = xraylib_coefficients(
    13, (30.0, 60.0, 100.0),
    kind="total-attenuation", density_g_cm3=2.6989,
)
material = aluminum.as_material_table()
print(material.linear_mm_inverse)
print(aluminum.provider)
```

The provider returns mass coefficients in cm²/g. DPT converts them to mm⁻¹
using `mu = coefficient * density / 10`. The provenance hash identifies the
canonical request and original mass outputs, together with the provider version
and Python/native implementation hashes. The native hash covers xraylib's
compiled reference data.

## Read an existing NIST response

`read_nist_coefficients` accepts the original response bytes and a `Provenance`
record with their SHA256, source and rights. Choose `format="xcom-html"` for
an XCOM result or `format="mass-attenuation-html"` for a NIST mass-attenuation
page. XCOM supplies interaction components and total attenuation with coherent
scattering. The separate mass-attenuation pages also supply energy absorption.

Choose XCOM's cm²/g output for interaction components. Some XCOM formats mix
barns/atom components with cm²/g totals. The reader checks the selected column's
units. It converts the decimal MeV energy into keV before rounding to binary64,
preserving the location of absorption edges.

## Resample a table

Adjacent duplicate energy nodes retain the below/above values at an absorption
edge. Every query chooses `edge_side="below"` or `"above"`. Between distinct
nodes, the table uses its declared linear or log-log interpolant.

`at_energies` returns linear coefficients in mm⁻¹. Its default support policy
rejects energies outside the supplied table. Choose `extrapolation="constant"`
to extend the outward endpoint or `"power-law"` to extend the nearest log-log
slope. A power-law continuation needs positive coefficients and a defined
endpoint slope. A duplicated endpoint encodes a jump and supplies no outside
slope.

## Download a reference file

The optional catalogue pins xraylib's elemental reference densities and BSD
notice to one upstream commit. Fetch it into a cache of your choosing:

```python
from dpt.reference_data import XRAYLIB_ELEMENTAL_DENSITIES, fetch_reference_data

reference = fetch_reference_data(
    XRAYLIB_ELEMENTAL_DENSITIES, "./reference-cache",
)
print(reference.data_path)
print(reference.licence_path)
print(reference.manifest_path)
```

The downloader verifies the licence before requesting the data, then verifies
both source hashes. A completed cache entry contains the original data, licence
and attribution. Subsequent calls verify and reuse that entry. A corrupt or
incomplete entry raises an error for you to inspect.

## Scientific references

- **Data:** [NIST XCOM](https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html), [xraylib](https://github.com/tschoonj/xraylib).
