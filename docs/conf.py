"""Sphinx configuration for Lit2Vec documentation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# -- Project information -----------------------------------------------------
project = "Lit2Vec"
copyright = f"{datetime.now().year}, Rahul Brahma"  # noqa: A001
author = "Rahul Brahma"
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("lit2vec")
except Exception:
    release = "0.0.0"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "myst_parser",
]

source_suffix = {
    ".rst": None,
    ".md": "markdown",
}

master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
pygments_style = "sphinx"

todo_include_todos = True

# -- Autodoc / Autosummary ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
# Heavy / GPU-only dependencies are mocked so the docs build without them.
autodoc_mock_imports = [
    "torch",
    "transformers",
    "accelerate",
    "xformers",
    "faiss",
    "faiss_cpu",
    "sentence_transformers",
    "annoy",
]

autosummary_generate = True
autosummary_imported_members = False

# -- Napoleon ----------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
}

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "attrs_inline",
    "fieldlist",
]
myst_heading_anchors = 3

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "Lit2Vec"
html_short_title = "Lit2Vec"
html_baseurl = "https://takshan.github.io/lit2vec"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_theme_options = {
    "logo": {
        "text": "Lit2Vec",
        "alt_text": "Lit2Vec - Embedding and search-index toolkit for biomedical literature",
    },
    "github_url": "https://github.com/takshan/lit2vec",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/lit2vec",
            "icon": "fab fa-python",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navbar_align": "content",
    "navbar_end": ["navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "navigation_depth": 4,
    "show_nav_level": 2,
    "collapse_navigation": False,
}

html_context = {
    "github_user": "takshan",
    "github_repo": "lit2vec",
    "github_version": "main",
    "doc_path": "docs",
}

html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs"],
}

html_show_sphinx = False
htmlhelp_basename = "Lit2Vecdoc"

# -- LaTeX output ------------------------------------------------------------
latex_elements = {}
latex_documents = [
    (master_doc, "Lit2Vec.tex", "Lit2Vec Documentation", author, "manual"),
]

# -- Manual page output ------------------------------------------------------
man_pages = [
    (master_doc, "lit2vec", "Lit2Vec Documentation", [author], 1),
]

# -- Texinfo output ----------------------------------------------------------
texinfo_documents = [
    (
        master_doc,
        "Lit2Vec",
        "Lit2Vec Documentation",
        author,
        "Lit2Vec",
        "Embedding and search-index toolkit for biomedical literature",
        "Miscellaneous",
    ),
]
