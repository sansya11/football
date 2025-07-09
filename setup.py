from setuptools import setup, find_packages

setup(
    name="football-player-reid",
    version="0.1.0",
    packages=find_packages(),
    author="Your Name",
    author_email="your.email@example.com",
    description="A football player re-identification system.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="<your-github-repo-url>",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
) 