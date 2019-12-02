FROM tiagopeixoto/graph-tool:latest
RUN pacman -S python-pip --noconfirm
# RUN pip install --upgrade pip
RUN pip install scanf numba

# to run jupyter notebook use docker run -p 8888:8888 -p 6006:6006 -it -u user -w /home/user beckman:v1 jupyter notebook --ip 0.0.0.0