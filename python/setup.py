from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parents[1]


class BuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        if compiler_type == "msvc":
            compile_args = ["/O2"]
        else:
            compile_args = ["-O3", "-std=c99"]

        for ext in self.extensions:
            ext.extra_compile_args = compile_args
        super().build_extensions()


setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=[
        Extension(
            "cb_sampler._backend",
            sources=["src/cb_sampler/_backend.c"],
            include_dirs=[str(ROOT / "src" / "c")],
            define_macros=[
                ("CB_ENABLE_CF", "0"),
                ("CB_ENABLE_ZIG", "1"),
                ("CB_ENABLE_DIAGNOSTICS", "0"),
            ],
        )
    ]
)
