#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
export PYTHONPATH=/agibot/data/home/agi/aimdk_msgs_pc2/local/lib/python3.10/dist-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/agibot/data/home/agi/aimdk_msgs_pc2/lib:${LD_LIBRARY_PATH:-}
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
exec /usr/bin/python3 /agibot/data/home/agi/x2-v02-state-executor-20261021.rMYM0G/x2_hal_readonly_probe.py --seconds 15 --tag soc1
