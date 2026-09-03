"""Route a --ct invocation to the shared real-data benchmark implementation."""

import sys


def real_ct_route(kind):
    if not any(value == "--ct" or value.startswith("--ct=") for value in sys.argv[1:]):
        return False
    if kind == "reference":
        from benchmarks.competitors import main
    else:
        if not any(value == "--cases" or value.startswith("--cases=") for value in sys.argv[1:]):
            sys.argv.extend(["--cases", "checks" if kind == "sampling" else "forward"])
        from benchmarks.real_ct import main
    main()
    return True
