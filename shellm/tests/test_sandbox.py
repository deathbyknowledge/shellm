import pytest
from shellm.sandbox import Sandbox


def test_sandbox_basic_functionality():
    """Test basic sandbox functionality including state persistence and error handling."""
    sandbox = Sandbox()
    sandbox.start()
    
    try:
        # Test state persistence
        stdout, stderr, exit_code = sandbox.execute_command("mkdir /workspace")
        assert exit_code == 0, f"mkdir failed: {stderr}"
        
        stdout, stderr, exit_code = sandbox.execute_command("cd /workspace")
        assert exit_code == 0, f"cd failed: {stderr}"
        
        stdout, stderr, exit_code = sandbox.execute_command("touch file.txt")
        assert exit_code == 0, f"touch failed: {stderr}"
        
        stdout, stderr, exit_code = sandbox.execute_command("ls -l")
        assert exit_code == 0, f"ls failed: {stderr}"
        assert "file.txt" in stdout, "Created file should be visible in ls output"
        
        # Test command failure
        stdout, stderr, exit_code = sandbox.execute_command("asdf")  # should fail
        assert exit_code != 0, "Invalid command should return non-zero exit code"
        
    finally:
        sandbox.stop() 