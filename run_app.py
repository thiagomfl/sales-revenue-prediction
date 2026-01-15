"""
Entry point for running the Streamlit app.
"""

import subprocess
import sys


def main() -> None:
  """
  Run the Streamlit application.
  """
  subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app/streamlit_app.py'], check=True)


if __name__ == '__main__':
  main()
