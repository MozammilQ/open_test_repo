FROM fedora:latest
RUN dnf upgrade -y
RUN dnf install -y python3-pip
RUN pip install jupyterlab
ENTRYPOINT ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--port=8080"]
