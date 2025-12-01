from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='segmentator',
    ext_modules=[
        CppExtension(
            name='segmentator.csrc.build.libsegmentator',
            sources=['csrc/segmentator.cpp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)


