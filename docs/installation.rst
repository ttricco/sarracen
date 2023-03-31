.. _installation:

============
Installation
============

Stable Release (Recommended)
----------------------------

The latest stable release and associated dependencies can be installed from `PyPi <https://pypi.org/project/sarracen/>`_:

.. code-block::

    pip install sarracen

This is the recommended way to install Sarracen.

Development Snapshot (via pip)
------------------------------

The latest development snapshot is available on the `GitHub repository <https://github.com/ttricco/sarracen>`_. This can
be installed directly through pip by using

.. code-block::

    pip install git+https://github.com/ttricco/sarracen.git

This will install Sarracen in your local environment, as well as any required dependencies.

Development Snapshot (via git)
------------------------------

Alternatively, the repository can be cloned locally and the dependencies installed manually. First, clone the
repository to your local machine either using the https link

.. code-block::

    git clone https://github.com/ttricco/sarracen.git

or clone through ssh using a password-protected ssh key

.. code-block::

    git clone git@github.com:ttricco/sarracen.git

The list of dependencies are stored in requirements.txt. These can be installed into your Python environment using pip

.. code-block::

    pip install -r requirements.txt

This method does not install Sarracen as a package in your Python environment. You will need to define the path to
Sarracen so that your Python code can locate it. See below.

.. code-block::

    import sys
    sys.path.append('/path/to/sarracen')  # replace with your path to Sarracen

    import sarracen

