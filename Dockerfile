FROM fedora:38
RUN dnf upgrade -y
RUN dnf install -y @development-tools rust cargo vim git cmake python3-devel python3-pip
RUN pip install jupyterlab \
                rustworkx>=0.13.0 \
                numpy>=1.17,<2 \
                scipy>=1.5 \
                sympy>=1.3 \
                dill>=0.3 \
                python-dateutil>=2.8.0 \
                stevedore>=3.0.0 \
                typing-extensions \
                symengine==0.9.2 \
                black[jupyter]~=22.0 \
                astroid==2.14.2 \
                pylint==2.16.2 \
                ruff==0.0.267 \
                coverage>=4.4.0 \
                hypothesis>=4.24.3 \
                stestr>=2.0.0,!=4.0.0 \
                ddt>=1.2.0,!=1.4.0,!=1.4.3 \
                Sphinx>=6.0,<7.2 \
                reno @git+https://github.com/openstack/reno.git@81587f616f17904336cdc431e25c42b46cd75b8f \
                sphinxcontrib-katex==0.9.9 

ENTRYPOINT ["jupyter", "lab", "--NotebookApp.token='8e34c90d7855df144bedcd8a1b5274769c7c11c86242ff66'", "--allow-root", "--ip=0.0.0.0", "--port=8080"]
