# Use an official Python base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgsl-dev \
    git \
    python3-dev \
    python3-pip

RUN apt-get update && apt-get install -y \
    gfortran \
    libhdf5-dev \
    libglpk-dev

RUN pip install jupyter notebook
RUN pip install tqdm
RUN pip install seaborn
RUN pip install scikit-learn
RUN pip install scikit-image
RUN pip install galpy
RUN pip install emcee
RUN pip install corner
RUN pip install astroquery
RUN pip install sklearn
RUN pip install dustmaps

    

# Create a working directory
WORKDIR /app

# Clone AGAMA source
RUN git clone https://github.com/GalacticDynamics-Oxford/Agama.git /app/Agama

# Install Python dependencies
RUN pip install numpy setuptools wheel scipy matplotlib pytest astropy numba pandas

# Switch to the AGAMA directory for installation
WORKDIR /app/Agama

# Set environment variable and install AGAMA
#RUN yes | pip install .
# Alternatively, use --config-settings if specific build options are needed
# RUN pip install --no-input --config-settings --build-option=--yes .

# Verify installation
#RUN python -c "import agama; print(agama.__version__)"

# Set the default command
CMD ["bash"]
