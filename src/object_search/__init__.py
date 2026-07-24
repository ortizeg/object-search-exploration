"""Exemplar-based object search.

Draw one box around an object in an image; find every other instance of that same object
in that same image. Four interchangeable search methods sit behind one ``SearchMethod``
protocol so the same query can be run through different algorithms and compared.

This is an exploration harness, not a product: each method is one self-contained,
top-to-bottom-readable module, and adding a fifth method is one new file plus one import.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
