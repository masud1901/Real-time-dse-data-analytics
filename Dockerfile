# Use the official Bitnami Spark image as the base
FROM bitnami/spark:3.3.0

# Switch to the root user to be able to install packages
USER root

# Copy the requirements file into the container's /tmp directory
COPY requirements.txt /tmp/

# Use pip to install the Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Switch back to the default non-root user (1001) for security
USER 1001 