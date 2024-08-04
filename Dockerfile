# Use the ubuntu base image
FROM ubuntu

# Set the working directory inside the container
WORKDIR /app

# Copy a script file to the container
COPY script.sh .

# Make the script eexecutable
RUN chmod +x script.sh

# Run the script when the container start
CMD [ "./scripts.sh" ]