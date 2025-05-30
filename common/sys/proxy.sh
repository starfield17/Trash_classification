#!/usr/bin/env bash

# Detect current shell
current_shell=$(ps -p $$ -ocomm=)

if [[ "$current_shell" == "fish" ]]; then
    # Execute proxy settings in fish subshell
    fish -c "
        read -P 'If you need to use a proxy, enter the proxy address (format: IP:port), otherwise press Enter to skip: ' proxy
        if [ -n \"\$proxy\" ]
            set -x http_proxy \"http://\$proxy\"
            set -x https_proxy \"https://\$proxy\"
            set -x ftp_proxy \"ftp://\$proxy\"
            set -x socks_proxy \"socks://\$proxy\"
            set -x no_proxy \"localhost,127.0.0.1,::1\"
            echo 'Proxy set to http://\$proxy'
            echo 'Proxy set to https://\$proxy'
            echo 'Proxy set to ftp://\$proxy'
            echo 'Proxy set to socks://\$proxy'
            echo 'no_proxy set to localhost,127.0.0.1,::1'
        else
            echo 'No proxy set, continuing...'
        end
    "
else
    # For bash and zsh
    read -p "If you need to use a proxy, enter the proxy address (format: IP:port), otherwise press Enter to skip: " proxy
    if [[ -n "$proxy" ]]; then
        export http_proxy="http://$proxy"
        export https_proxy="https://$proxy"
        export ftp_proxy="ftp://$proxy"
        export socks_proxy="socks://$proxy"
        export no_proxy="localhost,127.0.0.1,::1"
        echo "Proxy set to http://$proxy"
        echo "Proxy set to https://$proxy"
        echo "Proxy set to ftp://$proxy"
        echo "Proxy set to socks://$proxy"
        echo "no_proxy set to localhost,127.0.0.1,::1"
    else
        echo "No proxy set, continuing..."
    fi
fi
