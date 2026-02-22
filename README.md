# Python Refresher 2026

A comprehensive Python refresher project designed for TypeScript developers transitioning to Python's data science ecosystem. This project focuses on modern Python libraries with an emphasis on type safety, data validation, and interactive visualization.

## 🎯 Project Overview

This project serves as both a learning resource and practical reference for developers familiar with TypeScript who want to:
- Learn Python's data science stack
- Apply type safety principles to Python code
- Work with modern data validation libraries
- Create interactive visualizations
- Understand Python's ecosystem for TypeScript developers

## 📁 Project Structure

```
refresher2026/
├── src/                    # Source code examples
│   ├── test_pydantic.py   # Pydantic validation examples
│   ├── pandas_examples.py # Pandas DataFrame manipulation
│   ├── plotly.py          # Plotly visualization examples
│   └── basics.py          # Python basics (placeholder)
├── notebooks/             # Interactive learning notebooks
│   ├── basics.qmd        # Python basics and Pydantic
│   ├── pandas.qmd        # Pandas and Pandera
│   ├── plotly.qmd        # Plotly visualizations
│   └── *.html            # Exported notebook HTML files
├── python_refresher.md    # Comprehensive documentation
├── pyproject.toml        # Project configuration
├── uv.lock              # Dependency lock file
└── main.py              # Simple entry point
```

## 🛠️ Technologies

### Core Dependencies
- **Python 3.11** - Specified runtime version
- **Pydantic ≥2.12.5** - Runtime data validation (similar to Zod in TypeScript)
- **Pandas ≥3.0.0** - Data analysis and manipulation
- **Pandera ≥0.29.0** - Type checking for pandas DataFrames
- **Plotly[express] ≥6.5.2** - Interactive plotting library
- **NumPy ≥2.4.2** - Numerical computing

### Development Tools
- **UV** - Python package manager
- **Quarto** - Scientific publishing system for notebooks
- **Poethepoet** - Task runner
- **pandas-stubs** - Type stubs for pandas (for basedpyright type checking)

## 📚 Learning Materials

### 1. Comprehensive Documentation (`python_refresher.md`)
- Python basics for TypeScript developers
- Pydantic data validation patterns
- Pandas data manipulation techniques
- NumPy numerical computing
- Integration examples and comparisons

### 2. Interactive Notebooks (`notebooks/`)
- **Basics Notebook**: Python syntax, type hints, list comprehensions, Pydantic models
- **Pandas Notebook**: DataFrame creation, operations, Pandera type checking
- **Plotly Notebook**: Interactive visualizations with Iris and Tips datasets

### 3. Code Examples (`src/`)
- **Pydantic Examples**: Model validation, field validators, JSON parsing
- **Pandas Examples**: DataFrame creation from dict/JSON, statistical operations
- **Plotly Examples**: Box plots, scatter plots with color coding

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- UV package manager
- Quarto (for notebook rendering)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd refresher2026

# Install dependencies using UV
uv sync
```

### Running Examples
```bash
# Run Pydantic examples
python src/test_pydantic.py

# Run Pandas examples  
python src/pandas_examples.py

# Run Plotly examples
python src/plotly.py
```

### Working with Notebooks
```bash
# Render Quarto notebooks
quarto render notebooks/basics.qmd
quarto render notebooks/pandas.qmd  
quarto render notebooks/plotly.qmd

# Preview notebooks
quarto preview notebooks/basics.qmd
```

## 🎯 Key Features

### For TypeScript Developers
- Explicit syntax comparisons between Python and TypeScript
- Familiar patterns for data validation (Pydantic ≈ Zod)
- Type safety emphasis throughout the stack
- Modern tooling similar to npm/yarn ecosystem

### Data Science Focus
- Practical examples with real-world datasets
- Emphasis on data validation at multiple levels
- Interactive visualization capabilities
- Production-ready patterns for data pipelines

### Type Safety Layers
1. **Runtime Validation**: Pydantic models
2. **DataFrame Validation**: Pandera schemas  
3. **Static Type Checking**: Type hints + pandas-stubs
4. **Build-time Validation**: Basedpyright integration

## 📖 Learning Path

1. **Start with** `python_refresher.md` for conceptual understanding
2. **Explore** `notebooks/basics.qmd` for Python fundamentals
3. **Practice with** `src/` examples for hands-on coding
4. **Advance to** `notebooks/pandas.qmd` for data manipulation
5. **Visualize with** `notebooks/plotly.qmd` for interactive charts

## 🔧 Development

### Adding Dependencies
```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name
```

### Running Tasks
```bash
# Check available tasks (via poethepoet)
poe
```

### Type Checking
```bash
# Run type checking with basedpyright
basedpyright .
```

## 🤝 Contributing

This project is structured to be easily extended:
- Add new examples to `src/` directory
- Create new Quarto notebooks in `notebooks/`
- Update documentation in `python_refresher.md`
- Share TypeScript-to-Python patterns you discover

## 📄 License

This project is intended for educational purposes. Feel free to use and adapt the examples for your own learning.

## 🙏 Acknowledgments

- Built for TypeScript developers transitioning to Python
- Focuses on practical, production-ready patterns
- Emphasizes type safety in data science workflows
- Uses modern Python tooling (UV, Quarto, Pydantic v2)

---

**Happy Learning!** 🐍✨
