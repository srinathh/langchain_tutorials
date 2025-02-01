# LangChain Tutorials

Generic Minimal Data Science Application Template

## Features
* TODO

## Usage 

### Install package and application
```shell
pip install -e .
```

### To Run the application
```shell
langchain_tutorials
```

### Docker
```shell
docker build . -t langchain_tutorials
docker run langchain_tutorials
```

### To install with development dependencies
```shell
pip install -e ".[dev]"
```

### Changing Dependencies
- Make changes under `dependencies` in `pyproject.toml`

### Testing
```shell
pytest --cov
```

## Project Organization

## Project Organization

```
├── .gitignore                  <- to indicate which files to ignore by git
├── LICENSE                     <- Open-source license if one is chosen
├── README.md                   <- The top-level README for developers using this project.
├── data                        <- Folder for data. Any file inside this is ignored by Git
├── notebooks                   <- Jupyter notebooks. Suggested naming convention is a number (for ordering),
│                                  the creator's initials, and a short `-` delimited description, e.g.
│                                  `1.0-jqp-initial-data-exploration`.
├── Dockerfile                  <- Dockerfile to build the package with app as a Docker Image
├── pyproject.toml              <- Project configuration file with package metadata and dependency management
└── src                         <- Source code for use in this project.
    │
    ├── app 
    │   ├── __init__.py 
    │   └── app.py              <- contains the entry point for the sample Hello World application
    │
    ├── langchain_tutorials              <- the package for this project
    │   └── __init__.py 
    │
    └── utils 
        ├── __init__.py 
        ├── sample.py           <- A sample utility function example
        ├── singleton.py        <- A utility implementation of Singleton pattern for Python
        └── test_utils.py       <- An example of how to write tests
```

