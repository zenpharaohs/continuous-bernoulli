"""Compatibility entry point for exact endpoint qualification.

Run python3 tests/exact_endpoint_regression.py (NumPy and a C compiler).
The regular cb-validate command also runs this suite automatically.
"""
import pathlib
import runpy

if __name__ == '__main__':
    runpy.run_path(str(pathlib.Path(__file__).resolve().parents[1] /
                   'python/src/cb_sampler/endpoint_validation.py'), run_name='__main__')
