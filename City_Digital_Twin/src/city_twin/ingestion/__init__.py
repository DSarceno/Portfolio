"""Data Ingestion layer.

Responsibility: acquire the road network (OSMnx) and raw data, producing the directed
graph (with a stable, ordered edge index) and raw records for downstream layers.

Must NOT define mathematical or modeling assumptions. Depends on nothing upstream.
"""
