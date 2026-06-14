"""Database repository layer — encapsulates SQLAlchemy queries.

Each repository exposes only the methods its service needs, keeping
all ORM and query-builder imports private to the repository module.
"""
