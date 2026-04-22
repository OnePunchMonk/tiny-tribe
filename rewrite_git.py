import os
import subprocess

repo_dir = r'C:\Users\aggar\tiny-tribe'
os.chdir(repo_dir)

script = """
export GIT_AUTHOR_NAME="onepunchmonk"
export GIT_AUTHOR_EMAIL="aggarwal.avaya@yahoo.com"
export GIT_COMMITTER_NAME="onepunchmonk"
export GIT_COMMITTER_EMAIL="aggarwal.avaya@yahoo.com"
"""
subprocess.run(['git', 'filter-branch', '-f', '--env-filter', script, '--', '--all'])
