from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="CrowdFunds Raising Project",
    version="0.0.1",
    author="Mani Krishna",
    author_email="mandepudi.mk@gmail.com",
    description="Crowd Fund Raising Project",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
