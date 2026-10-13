"""Training: option resolution, preflight checks, provenance, Ultralytics runner.

Ultralytics is imported lazily inside ``firewatch.training.runner`` so the
options and checks can be used (and tested) without torch installed.
"""
