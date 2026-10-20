"""FireWatch video-analysis application: inference, temporal events, job API and dashboard.

Pure-Python modules (config, events, store, jobs, server) import nothing heavy; OpenCV, NumPy and
Ultralytics are imported lazily by ``sources``, ``annotate`` and ``detector`` when a video is
actually processed.
"""
