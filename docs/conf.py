# -*- coding: utf-8 -*-
#
# Nalu-Wind documentation build configuration file, created by
# sphinx-quickstart on Wed Jan 25 13:52:07 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import subprocess
import re

#sys.path.append(os.path.abspath('_extensions/'))

readTheDocs = os.environ.get('READTHEDOCS', None) == 'True'
# Only link to API docs if the user specifically requests it. On RTD build it by default
use_breathe = tags.has("use_breathe") or readTheDocs

if readTheDocs:
    sourcedir = sys.argv[-2]
    builddir = sys.argv[-1]
elif use_breathe:
    sourcedir = sys.argv[-6]
    builddir = sys.argv[-5]
else:
    sourcedir = sys.argv[-2]
    builddir = sys.argv[-1]


# This function was adapted from https://gitlab.kitware.com/cmb/smtk
# Only run when on readthedocs
def runDoxygen(doxyfileIn, doxyfileOut):
    dx = open(os.path.join(sourcedir, doxyfileIn), 'r')
    cfg = dx.read()
    srcdir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    bindir = srcdir
    c2 = re.sub('@CMAKE_SOURCE_DIR@', srcdir,
                re.sub('@CMAKE_BINARY_DIR@', bindir, cfg))
    doxname = os.path.join(sourcedir, doxyfileOut)
    with open(doxname, 'w') as fh:
        fh.write(c2)
    print('Running Doxygen on %s' % doxyfileOut)
    try:
        subprocess.call(('doxygen', doxname))
    except:
        # Gracefully bailout if doxygen encounters errors
        use_breathe = False

if readTheDocs:
    runDoxygen('Doxyfile.breathe.in', 'Doxyfile')

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.5.2'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.mathjax',
              'sphinx.ext.intersphinx',
              'sphinxcontrib.bibtex',
             ]
bibtex_bibfiles = ['references/references.bib']

if use_breathe:
    extensions.append('breathe')

autodoc_default_flags = ['members','show-inheritance','undoc-members']

autoclass_content = 'both'

mathjax_path = 'https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'


# Add any paths that contain templates here, relative to this directory.
#templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Nalu-Wind'
copyright = u'2019, Nalu-Wind Development Team'
author = u'Nalu-Wind Team'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = u'1.2.0'
# The full version, including alpha/beta/rc tags.
release = u'1.2.0'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None
numfig = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'bizstyle'
if readTheDocs:
    html_theme = 'default'
else:
    html_theme = 'sphinx_rtd_theme'
html_logo = 'naluLowMach.jpg'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'Naludoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'Nalu.tex', u'Nalu-Wind Documentation',
     u'Nalu-Wind Development Team', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'nalu', u'Nalu-Wind Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'Nalu', u'Nalu-Wind Documentation',
     author, 'Nalu', 'One line description of project.',
     'Miscellaneous'),
]

# Breathe options
breathe_projects = {
    'nalu' : os.path.join(sourcedir if readTheDocs else builddir,
                          'doxygen', 'xml'),
    'example_cpp' : os.path.join(sourcedir,
                                 'source', 'developer', 'dox_example', 'xml')
}

# Assign nalu to be the default project
breathe_default_project = "nalu"

# Set primary language to C++ for documentation instead of default `py:`
primary_domain = "cpp"

def setup(app):
    app.add_object_type("inpfile", "inpfile",
                        objname="Nalu-Wind Input File Parameter",
                        indextemplate="pair: %s; Nalu-Wind Input File Parameter")

    app.add_config_value("use_breathe", use_breathe, 'env')
    app.add_config_value("on_rtd", readTheDocs, 'env')
