#!/bin/bash
#SBATCH --job-name=ariel_run
#SBATCH --output=slurm_terminal.txt
#SBATCH --error=slurm_terminal.txt
#SBATCH --time=00:00:10

# Docker container configurable arguments
IMAGE_NAME="ariel_unet2:latest"
CONTAINER_NAME="ariel_run_slurm"
PROJECT_DIR="/home/dsi/arielro1/tmp/rfproj"
SCRIPT_COMMAND="python test.py"
TIME_TO_RUN=10  # job timelimit

# start container in detached mode
docker run --init -d --rm \
  --name "$CONTAINER_NAME" \
  --net host \
  --gpus all \
  -v "$PROJECT_DIR":/app \
  "$IMAGE_NAME" \
  bash -c "source ~/.bashrc && PYTHONUNBUFFERED=1 $SCRIPT_COMMAND" > /dev/null

echo "$CONTAINER_NAME terminal output:"
# output container logs live (optinal)
docker logs -f "$CONTAINER_NAME" &

# background timer to send SIGTERM if timeout hits
(
  sleep "$TIME_TO_RUN"
  if docker ps -q -f name="$CONTAINER_NAME" > /dev/null; then
    echo "Timeout reached. Sending SIGTERM to container..."
    docker exec "$CONTAINER_NAME" kill -TERM 1
  fi
) &

# wait for container to finish naturally
docker wait "$CONTAINER_NAME"
