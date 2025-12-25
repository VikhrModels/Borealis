#!/bin/bash

# Borealis Instruct Training Script
# Launches training in a screen session

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv/bin/activate"
CONFIG_PATH="$SCRIPT_DIR/configs/Borealis_5B_instruct.yaml"
SCREEN_NAME="borealis_train"

# Check if screen session already exists
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "Screen session '$SCREEN_NAME' already exists!"
    echo "Use: screen -r $SCREEN_NAME to attach"
    echo "Or kill it first: screen -S $SCREEN_NAME -X quit"
    exit 1
fi

# Launch training in screen
screen -dmS "$SCREEN_NAME" bash -c "
    source $VENV_PATH
    export HF_AUDIO_DECODER_BACKEND=soundfile
    cd $SCRIPT_DIR
    accelerate launch --config_file accelerate_config.yaml train_instruct.py --config $CONFIG_PATH
    echo 'Training finished. Press any key to exit.'
    read
"

echo "Training started in screen session: $SCREEN_NAME"
echo "Attach with: screen -r $SCREEN_NAME"
echo "Detach with: Ctrl+A, D"
