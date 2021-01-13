import pathlib

from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="coinstac-dinunet",
    version="0.2.1",
    description="Distributed Neural Network implementation on COINSTAC.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/trendscenter/coinstac-dinunet",
    author="Aashis Khana1",
    author_email="sraashis@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=['coinstac_dinunet', 'coinstac_dinunet.config', 'coinstac_dinunet.data',
              'coinstac_dinunet.metrics', 'coinstac_dinunet.nodes',
              'coinstac_dinunet.utils', 'coinstac_dinunet.vision'],
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'scikit-learn', 'scikit-image',
                      'pillow', 'matplotlib', 'opencv-python', 'pandas', 'seaborn']
)
