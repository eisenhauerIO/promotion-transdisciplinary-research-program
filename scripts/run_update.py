#!/usr/bin/env python
""" This script updates all files, including the submodules.
"""
import subprocess

cmds = list()
cmds += [["git", "pull"]]
cmds += [["git", "submodule", "update", "--init", "--recursive", "--remote"]]
[subprocess.check_call(cmd) for cmd in cmds]
