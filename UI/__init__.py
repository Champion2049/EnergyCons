"""UI package initializer for Flask application deployment.

Allows importing the Flask app object as:

    from UI.app import app  # original module
or
    from UI import app  # because we re-export below

Used primarily for WSGI servers (e.g., PythonAnywhere) that expect a top-level
`application` callable which you can map to `app`.
"""

from .app import app  # re-export

__all__ = ["app"]
