FROM fedora:38
RUN dnf upgrade -y
RUN dnf install -y @development-tools rust cargo vim git cmake python3-devel python3-pip
RUN pip install 'jupyterlab'
RUN pip install 'rustworkx>=0.13.0'
RUN pip install 'numpy>=1.17,<2'
RUN pip install 'scipy>=1.5'
RUN pip install 'sympy>=1.3'
RUN pip install 'dill>=0.3'
RUN pip install 'python-dateutil>=2.8.0'
RUN pip install 'stevedore>=3.0.0'
RUN pip install 'typing-extensions'
RUN pip install 'symengine==0.9.2'
RUN pip install 'black[jupyter]~=22.0'
RUN pip install 'astroid==2.14.2'
RUN pip install 'pylint==2.16.2'
RUN pip install 'ruff==0.0.267'
RUN pip install 'coverage>=4.4.0'
RUN pip install 'hypothesis>=4.24.3'
RUN pip install 'stestr>=2.0.0,!=4.0.0'
RUN pip install 'ddt>=1.2.0,!=1.4.0,!=1.4.3'
RUN pip install 'Sphinx>=6.0,<7.2'
RUN pip install 'reno @git+https://github.com/openstack/reno.git@81587f616f17904336cdc431e25c42b46cd75b8f'
RUN pip install 'sphinxcontrib-katex==0.9.9'

CMD ["jupyter", "lab", "--NotebookApp.token='8e34c90d7855df144bedcd8a1b5274769c7c11c86242ff66'", "--allow-root", "--ip=0.0.0.0", "--port=8080"]
  
