#!/bin/bash

# add_to_systemd.sh
# Usage:
#   Install service: bash add_to_systemd.sh install "conda_env_name" "/path/to/script.py"
#   Remove service:  bash add_to_systemd.sh remove "/path/to/script.py"

echo "==============================="
echo "Initialize startup script: add_to_systemd.sh"
echo "==============================="

# Make sure at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage:"
    echo "  Install service: $0 install \"conda_env_name\" \"/path/to/script.py\""
    echo "  Remove service:  $0 remove \"/path/to/script.py\""
    exit 1
fi

ACTION=$1

# Function: install service
install_service() {
    if [ "$#" -ne 3 ]; then
        echo "Error: Installing a service requires three arguments."
        echo "Usage: $0 install \"conda_env_name\" \"/path/to/script.py\""
        exit 1
    fi

    CONDA_ENV=$2
    SCRIPT_PATH=$3

    # Convert to absolute path
    SCRIPT_PATH=$(realpath "$SCRIPT_PATH")

    echo "Conda environment: ${CONDA_ENV:-'System default Python environment'}"
    echo "Python script path: $SCRIPT_PATH"

    # Check that the script exists
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "Error: Script file '$SCRIPT_PATH' does not exist."
        exit 1
    fi

    # Get current user information
    USER_NAME=$(whoami)
    USER_HOME=$(eval echo "~$USER_NAME")

    echo "Current user: $USER_NAME"
    echo "User home directory: $USER_HOME"

    # Define service name
    SERVICE_NAME=$(basename "$SCRIPT_PATH" .py).service

    echo "systemd service to be created: $SERVICE_NAME"

    # Check if a service with the same name already exists
    if systemctl list-unit-files | grep -q "^$SERVICE_NAME"; then
        echo "Warning: Service '$SERVICE_NAME' already exists. It will be overwritten."
    fi

    # Locate the Conda installation path
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -z "$CONDA_BASE" ]; then
        echo "Error: Conda is not installed or not found in PATH."
        exit 1
    fi

    echo "Conda base path: $CONDA_BASE"

    # Build the ExecStart command
    if [ -n "$CONDA_ENV" ]; then
        EXEC_START_CMD="/bin/bash -c 'echo \"Activating Conda environment: $CONDA_ENV\"; source \"$CONDA_BASE/etc/profile.d/conda.sh\"; conda activate \"$CONDA_ENV\"; if [ \"\$?\" -ne 0 ]; then echo \"Error: Could not activate Conda environment '$CONDA_ENV'.\"; exit 1; fi; echo \"Starting Python script: $SCRIPT_PATH\"; python \"$SCRIPT_PATH\"'"
    else
        EXEC_START_CMD="/bin/bash -c 'echo \"Using system default Python environment.\"; echo \"Starting Python script: $SCRIPT_PATH\"; python \"$SCRIPT_PATH\"'"
    fi

    # Create the systemd service file content
    SERVICE_FILE="[Unit]
Description=Auto-start $SERVICE_NAME
After=network.target

[Service]
Type=simple
User=$USER_NAME
Environment=PATH=$CONDA_BASE/bin:/usr/bin:/bin:/usr/local/bin
WorkingDirectory=$(dirname "$SCRIPT_PATH")
ExecStart=$EXEC_START_CMD

Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target"

    echo "Creating systemd service file..."

    # Write the service file to /etc/systemd/system/
    echo "$SERVICE_FILE" | sudo tee /etc/systemd/system/$SERVICE_NAME > /dev/null

    if [ "$?" -ne 0 ]; then
        echo "Error: Unable to write service file to /etc/systemd/system/$SERVICE_NAME. Check permissions."
        exit 1
    fi

    echo "Service file created: /etc/systemd/system/$SERVICE_NAME"

    # Reload systemd daemon
    echo "Reloading systemd daemon..."
    sudo systemctl daemon-reload

    if [ "$?" -ne 0 ]; then
        echo "Error: Failed to reload systemd daemon."
        exit 1
    fi

    # Enable the service to start at boot
    echo "Enabling service '$SERVICE_NAME'..."
    sudo systemctl enable $SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "Error: Failed to enable service '$SERVICE_NAME'."
        exit 1
    fi

    # Start the service
    echo "Starting service '$SERVICE_NAME'..."
    sudo systemctl start $SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "Error: Failed to start service '$SERVICE_NAME'."
        exit 1
    fi

    echo "=========================================="
    echo "Service '$SERVICE_NAME' has been created and started successfully."
    echo "It will run automatically at system startup."
    echo "=========================================="

    echo "You can check the service status with:"
    echo "  systemctl status $SERVICE_NAME"
    echo "View service logs with:"
    echo "  journalctl -u $SERVICE_NAME -f"
}

# Function: remove service
remove_service() {
    if [ "$#" -ne 2 ]; then
        echo "Error: Removing a service requires two arguments."
        echo "Usage: $0 remove \"/path/to/script.py\""
        exit 1
    fi

    SCRIPT_PATH=$2

    # Convert to absolute path
    SCRIPT_PATH=$(realpath "$SCRIPT_PATH")

    echo "Python script path: $SCRIPT_PATH"

    # Define service name
    SERVICE_NAME=$(basename "$SCRIPT_PATH" .py).service

    echo "systemd service to be removed: $SERVICE_NAME"

    # Check if the service exists
    if ! systemctl list-unit-files | grep -q "^$SERVICE_NAME"; then
        echo "Error: Service '$SERVICE_NAME' does not exist."
        exit 1
    fi

    # Stop the service
    echo "Stopping service '$SERVICE_NAME'..."
    sudo systemctl stop $SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "Warning: Failed to stop service '$SERVICE_NAME'. It may already be stopped."
    else
        echo "Service '$SERVICE_NAME' has been stopped."
    fi

    # Disable the service
    echo "Disabling service '$SERVICE_NAME'..."
    sudo systemctl disable $SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "Warning: Failed to disable service '$SERVICE_NAME'."
    else
        echo "Service '$SERVICE_NAME' has been disabled."
    fi

    # Delete the service file
    echo "Removing service file '/etc/systemd/system/$SERVICE_NAME'..."
    sudo rm -f /etc/systemd/system/$SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "Error: Could not remove service file '/etc/systemd/system/$SERVICE_NAME'."
        exit 1
    fi

    # Reload systemd daemon
    echo "Reloading systemd daemon..."
    sudo systemctl daemon-reload

    if [ "$?" -ne 0 ]; then
        echo "Error: Failed to reload systemd daemon."
        exit 1
    fi

    echo "=========================================="
    echo "Service '$SERVICE_NAME' has been removed successfully."
    echo "=========================================="

    echo "You can verify that the service has been removed with:"
    echo "  systemctl status $SERVICE_NAME"
}

# Call the appropriate function based on ACTION
case "$ACTION" in
    install)
        install_service "$@"
        ;;
    remove)
        remove_service "$@"
        ;;
    *)
        echo "Error: Invalid operation '$ACTION'."
        echo "Supported operations: install, remove"
        exit 1
        ;;
esac
