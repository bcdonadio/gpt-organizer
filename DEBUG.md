# Debugging Setup for ChatGPT Organizer

This project has been configured with `debugpy` for comprehensive debugging capabilities both in VSCode and via manual/programmatic debugging.

## Installation

The `debugpy` package has been added as a development dependency in `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    # ... other dependencies
    "debugpy>=1.8.0",
]
```

To install all dependencies including debugpy:

```bash
uv sync --group dev
```

## VSCode Debugging Configurations

The project includes pre-configured launch configurations in [`.vscode/launch.json`](.vscode/launch.json):

### Available Configurations

1. **Python: Current File**
   - Debug the currently open Python file
   - Uses integrated terminal
   - Loads environment variables from `.env`

2. **Python: Main**
   - Debug the main entry point ([`main.py`](main.py))
   - Pre-configured for the project's main script

3. **Python: Categorize Module**
   - Debug the GptCategorize module
   - Runs as a module (`python -m GptCategorize`)

4. **Python: Move Module**
   - Debug the GptMove module
   - Runs as a module (`python -m GptMove`)

5. **Python: Remote Attach**
   - Attach to a remote debugpy server
   - Default: `localhost:5678`
   - Use with manual debugging setup

6. **Python: Attach to Process**
   - Attach to a running Python process
   - VSCode will prompt for process selection

### Using VSCode Debugging

1. **Set Breakpoints**: Click in the gutter next to line numbers
2. **Start Debugging**:
   - Press `F5` or use the Run panel
   - Select desired configuration from dropdown
3. **Debug Controls**:
   - `F5`: Continue
   - `F10`: Step Over
   - `F11`: Step Into
   - `Shift+F11`: Step Out
   - `Shift+F5`: Stop

## Manual/Programmatic Debugging

The project includes a [`debug_config.py`](debug_config.py) module for advanced debugging scenarios.

### Basic Usage

```python
from debug_config import start_debug_server, debug_here

# Start debug server (non-blocking)
start_debug_server()

# Or start and wait for debugger to attach
start_debug_server(wait_for_client=True)

# Debug at specific location in code
debug_here()  # Will pause execution here when debugger attaches
```

### Available Functions

#### `start_debug_server(host="localhost", port=5678, wait_for_client=False, log_to_stderr=False)`

Starts the debugpy server for remote attachment.

**Parameters:**

- `host`: Host to bind to (default: localhost)
- `port`: Port to bind to (default: 5678)
- `wait_for_client`: Block until debugger attaches
- `log_to_stderr`: Enable debug logging

#### `debug_here(host="localhost", port=5678)`

Starts debugging at the current code location.

#### `is_debugger_attached()`

Returns `True` if a debugger is currently attached.

#### `enable_debugging_on_exception()`

Automatically start debugging when unhandled exceptions occur.

### Environment Variable Configuration

You can enable debugging automatically using environment variables:

```bash
# Enable debugging
export DEBUGPY_ENABLE=1

# Optional: Configure host/port
export DEBUGPY_HOST=localhost
export DEBUGPY_PORT=5678

# Wait for debugger to attach before continuing
export DEBUGPY_WAIT=1

# Run your script
python main.py --conversations-json conversations.json
```

**Environment Variables:**

- `DEBUGPY_ENABLE`: Set to `1`, `true`, or `yes` to enable
- `DEBUGPY_HOST`: Host to bind to (default: localhost)
- `DEBUGPY_PORT`: Port to bind to (default: 5678)
- `DEBUGPY_WAIT`: Set to `1`, `true`, or `yes` to wait for client

## Testing the Setup

Use the included test script to verify everything is working:

```bash
# Run the test script
uv run python test_debug.py

# Test environment variable debugging
DEBUGPY_ENABLE=1 DEBUGPY_WAIT=1 uv run python test_debug.py
```

The test script will:

- Verify debugpy installation
- Test debug server startup
- Demonstrate remote attachment
- Show environment variable usage

## Common Debugging Workflows

### 1. Debug Main Application

```bash
# Start with VSCode
# 1. Open main.py
# 2. Set breakpoints
# 3. Press F5 and select "Python: Main"

# Or debug with arguments
# Use "Python: Main" configuration and modify args in launch.json
```

### 2. Remote Debugging

```bash
# Terminal 1: Start your script with debug server
DEBUGPY_ENABLE=1 DEBUGPY_WAIT=1 uv run python main.py --conversations-json conversations.json

# Terminal 2 or VSCode: Attach debugger
# Use "Python: Remote Attach" configuration
```

### 3. Debug on Exception

```python
# Add to your script
from debug_config import enable_debugging_on_exception
enable_debugging_on_exception()

# Now any unhandled exception will start the debug server
```

### 4. Conditional Debugging

```python
from debug_config import debug_here

def process_data(data):
    if some_error_condition:
        debug_here()  # Only debug when needed
    # ... rest of function
```

## Advanced Usage

### Custom Debug Integration

```python
import os
from debug_config import start_debug_server, is_debugger_attached

class MyApplication:
    def __init__(self):
        # Enable debugging in development
        if os.environ.get('DEBUG_MODE'):
            start_debug_server(wait_for_client=True)

    def process(self):
        if is_debugger_attached():
            print("Debugger is attached - detailed logging enabled")
        # ... processing logic
```

### Debugging Multiple Processes

```python
import multiprocessing
from debug_config import start_debug_server

def worker_process(port_offset):
    # Each process gets its own debug port
    start_debug_server(port=5678 + port_offset)
    # ... worker logic

# Start multiple workers with different debug ports
for i in range(3):
    p = multiprocessing.Process(target=worker_process, args=(i,))
    p.start()
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**

   ```text
   Error: Address already in use
   ```

   - Try different port: `start_debug_server(port=5679)`
   - Check for running debug sessions: `ps aux | grep debugpy`

2. **Debugger Won't Attach**
   - Ensure firewall allows port 5678
   - Verify host/port settings match
   - Check VSCode Python extension is installed

3. **Breakpoints Not Hit**
   - Ensure `justMyCode: false` in launch.json
   - Verify file paths are correct
   - Check if code is actually executed

4. **Environment Variables Not Working**
   - Verify environment variable names and values
   - Check if variables are exported properly
   - Use `printenv | grep DEBUGPY` to verify

### Debug Output

Enable verbose logging:

```python
from debug_config import start_debug_server
start_debug_server(log_to_stderr=True)
```

## Security Considerations

- Debug server listens on localhost by default (secure)
- Never expose debug ports to external networks in production
- Disable debugging in production environments
- Be cautious with `debug_here()` in production code

## Performance Impact

- Minimal overhead when debugger is not attached
- Some overhead when debugger is attached
- Environment variable checking happens at import time
- Use conditional debugging for production deployments

## Integration with CI/CD

```yaml
# Example GitHub Actions workflow
name: Debug Tests
on: [push, pull_request]
jobs:
  test-debug:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.13'
      - name: Install dependencies
        run: uv sync --group dev
      - name: Test debug configuration
        run: uv run python test_debug.py
```

## Resources

- [debugpy Documentation](https://github.com/microsoft/debugpy)
- [VSCode Python Debugging](https://code.visualstudio.com/docs/python/debugging)
- [Python Debugging Guide](https://docs.python.org/3/library/pdb.html)
