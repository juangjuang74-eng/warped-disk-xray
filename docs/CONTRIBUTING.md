# Contributing

## Project structure rules

- One research gap = one module in `src/`
- Every module has a CLI entry point (`if __name__ == "__main__"`)
- Every new function in `src/utils/physics.py` needs a test in `tests/test_physics.py`
- Data never goes into git — use `data/` locally and document in `docs/data_formats.md`

## Adding a new research gap

1. Create `src/your_module/` with `__init__.py` and main script
2. Add a config YAML in `configs/` if the module has hyperparameters
3. Add a notebook in `notebooks/` demonstrating usage
4. Update the gap table in `README.md`

## Running tests

```bash
pytest tests/ -v
```

## Code style

```bash
black src/ tests/
```
