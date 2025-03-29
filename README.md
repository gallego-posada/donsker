# donsker
Simulations for talk on Donsker's Theorem

### Getting started
```bash
# Install uv, select Python 3.12 and create a virtual environment
brew install uv
uv python install 3.12
uv init
```

### Running scripts
```bash
uv python <script_name>.py

# Several scripts use command line arguments. For example,
uv run bridge.py --keep_open --num_frames 5 --sample_size 100
```

### Code formatting

```bash
uvx ruff check . --extend-select I --fix; uvx ruff format . --line-length=120
```