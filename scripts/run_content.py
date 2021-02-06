#!/usr/bin/env python
"""This script compiles the paper."""
import argparse
import shutil
import glob
import os

ROOT = os.environ["PROJECT_ROOT"]


def compile_latex_document(dirname=None):
    cwd = os.getcwd()
    if dirname:
        os.chdir(dirname)

    [
        os.system(type_ + " main")
        for type_ in ["pdflatex", "bibtex", "pdflatex", "pdflatex"]
    ]

    os.chdir(cwd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Create content")

    parser.add_argument("-p", "--paper", action="store_true", help="compile paper")

    parser.add_argument("-s", "--slides", action="store_true", help="compile slides")

    args = parser.parse_args()

    os.chdir(ROOT)

    if args.paper:
        compile_latex_document(ROOT + "/promotion/paper")
        shutil.copy("promotion/paper/main.pdf", "manuscript.pdf")

    if args.slides:
        compile_latex_document(ROOT + "/promotion/slides")
        shutil.copy("promotion/slides/main.pdf", "slides.pdf")
