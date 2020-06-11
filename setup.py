from setuptools import setup

with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name="legitindicators",
    version="0.0.33",
    description="Legit indicators to be used in trading strategies.",
    py_modules=["legitindicators"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = ['numpy',],
    extras_require = {
        "dev" : [
            "pytest>=3.7",
        ],
    },
    url="https://github.com/aticio/legitindicators",
    author="Özgür Atıcı",
    author_email="aticiozgur@gmail.com",
)