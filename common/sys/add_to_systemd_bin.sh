#!/bin/bash

# add_to_systemd_bin.sh
# Usage:
#   Install service: bash add_to_systemd_bin.sh install "/path/to/executable"
#   Remove service: bash add_to_systemd_bin.sh remove "/path/to/executable"

echo "==============================="
echo "Initializing startup script: add_to_systemd_bin.sh"
echo "==============================="

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Error: Incorrect number of parameters."
    echo "Usage:"
    echo "  Install service: $0 install \"/path/to/executable\""
    echo "  Remove service: $0 remove \"/path/to/executable\""
    exit 1
fi

ACTION=$1

# Function: Install service
install_service() {
    if [ "$#" -ne 2 ]; then
        echo "Error: Installing service requires two parameters."
        echo "Usage: $0 install \"/path/to/executable\""
        exit 1
    fi

    EXECUTABLE_PATH=$2

    # Convert to absolute path
    EXECUTABLE_PATH=$(realpath "$EXECUTABLE_PATH")

    echo "Executable path: $EXECUTABLE_PATH"

    # Check if executable file exists
    if [ ! -f "$EXECUTABLE_PATH" ]; then
        echo "Error: Executable file '$EXECUTABLE_PATH' does not exist."
        exit 1
    fi

    # Check if file has execute permission
    if [ ! -x "$EXECUTABLE_PATH" ]; then
        echo "Warning: '$EXECUTABLE_PATH' lacks execute permission. Adding execute permission..."
        chmod +x "$EXECUTABLE_PATH"
        if [ "$?" -ne 0 ]; then
            echo "Error: Failed to add execute permission to '$EXECUTABLE_PATH'."
            exit 1
        fi
    fi

    # Get current user information
    USER_NAME=$(whoami)
    USER_HOME=$(eval echo "~$USER_NAME")

    echo "Current user: $USER_NAME"
    echo "User home directory: $USER_HOME"

    # Define service name (using filename as service name)
    SERVICE_NAME=$(basename "$EXECUTABLE_PATH").service

    echo "Will create systemd service name: $SERVICE_NAME"

    # Check if service with same name already exists
    if systemctl list-unit-files | grep -q "^$SERVICE_NAME"; then
        echo "Warning: Service '$SERVICE_NAME' already exists. Preparing to overwrite."
    fi

    # Create systemd service file content
    SERVICE_FILE="[Unit]
Description=Auto-start $SERVICE_NAME
After=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$(dirname "$EXECUTABLE_PATH")
ExecStart=$EXECUTABLE_PATH
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target"

    echo "Creating systemd service file..."

    # Write service file to /etc/systemd/system/
    echo "$SERVICE_FILE" | sudo tee /etc/systemd/system/$SERVICE_NAME > /dev/null

    if [ "$?" -ne 0 ]; then
        echo "Error: Could not write service file to /etc/systemd/system/$SERVICE_NAME. Please check permissions."
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

    # Enable service to start at boot
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
    echo "Service '$SERVICE_NAME' has been successfully created and started."
    echo "It will automatically run at system startup."
    echo "==========================================""

    echo "You can use the following commands to check service status:"
    echo "  systemctl status $SERVICE_NAME"
    echo "To view service logs use:"
    echo "  journalctl -u $SERVICE_NAME -f"
}

# Function: Remove service
remove_service() {
    if [ "$#" -ne 2 ]; then
        echo "Error: Removing service requires two parameters."
        echo "Usage: $0 remove \"/path/to/executable\""
        exit 1
    fi

    EXECUTABLE_PATH=$2

    # Convert to absolute path
    EXECUTABLE_PATH=$(realpath "$EXECUTABLE_PATH")

    echo "Executable path: $EXECUTABLE_PATH"

    # Define service name
    SERVICE_NAME=$(basename "$EXECUTABLE_PATH").service

    echo "Will remove systemd service name: $SERVICE_NAME"

    # Check if service exists
    if ! systemctl list-unit-files | grep -q "^$SERVICE_NAME"; then
        echo "Error: Service '$SERVICE_NAME' does not exist."
        exit 1
    fi

    # Stop service
    echo "Stopping service '$SERVICE_NAME'..."
    sudo systemctl stop $SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "Warning: Failed to stop service '$SERVICE_NAME'. Service might already be stopped."
    else
        echo "Service '$SERVICE_NAME' has been stopped."
    fi

    # Disable service
    echo "Disabling service '$SERVICE_NAME'..."
    sudo systemctl disable $SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "Warning: Failed to disable service '$SERVICE_NAME'."
    else
        echo "Service '$SERVICE_NAME' has been disabled."
    fi

    # Delete service file
    echo "Deleting service file '/etc/systemd/system/$SERVICE_NAME'..."
    sudo rm -f /etc/systemd/system/$SERVICE_NAME

    if [ "$?" -ne 0 ]; then
        echo "Error: Could not delete service file '/etc/systemd/system/$SERVICE_NAME'."
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
    echo "Service '$SERVICE_NAME' has been successfully removed."
    echo "==========================================""

    echo "You can use the following commands to verify service removal:"
    echo "  systemctl status $SERVICE_NAME"
}

# Call appropriate function based on ACTION
case "$ACTION" in
    install)
        install_service "$@"
        ;;
    remove)
        remove_service "$@"
        ;;
    *)
        echo "Error: Invalid action '$ACTION'."
        echo "Supported actions: install, remove"
        exit 1
        ;;
esac
