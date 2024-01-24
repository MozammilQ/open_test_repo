FROM fedora:latest
RUN dnf upgrade -y
RUN dnf install -y python3-pip
RUN pip install jupyterlab
ENTRYPOINT ["jupyter", "lab", "--NotebookApp.token='8e34c90d7855df144bedcd8a1b5274769c7c11c86242ff66'", "--allow-root", "--ip=0.0.0.0", "--port=8080"]
