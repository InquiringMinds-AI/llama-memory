# Contributing to llama-memory

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone
git clone https://github.com/InquiringMinds-AI/llama-memory.git
cd llama-memory

# Install dependencies
./scripts/install.sh

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Code Style

We use:
- [Black](https://github.com/psf/black) for formatting
- [Ruff](https://github.com/astral-sh/ruff) for linting

```bash
black llama_memory/
ruff check llama_memory/
```

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Format code (`black .`)
6. Commit (`git commit -m 'Add amazing feature'`)
7. Push (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Reporting Issues

Please include:
- Platform (Linux/macOS/Android)
- Architecture (x86_64/aarch64)
- Python version
- Output of `llama-memory doctor`
- Steps to reproduce

## Areas for Contribution

- Additional embedding model support
- Performance optimizations
- Platform-specific fixes
- Documentation improvements
- Test coverage

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
