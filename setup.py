import setuptools

try:
   import pypandoc
   description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   description = open('README.md').read()

setuptools.setup(
    packages = setuptools.find_packages(),

    install_requires = ['numpy', 'scipy', 'mako', 'cppimport', 'joblib', 'pyopencl'],
    zip_safe = False,
    include_package_data = True,

    name = 'tectosaur',
    version = '0.0.1',
    description = 'Boundary element methods for crustal deformation and earthquake science.',
    long_description = description,

    url = 'https://github.com/tbenthompson/tectosaur',
    author = 'T. Ben Thompson',
    author_email = 't.ben.thompson@gmail.com',
    license = 'MIT',
    platforms = ['any'],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Operating System :: POSIX',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++'
    ]
)
