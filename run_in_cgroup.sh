#!/bin/bash
# Filename: run_in_cgroup.sh

# 1. 找到自己所在的 cgroup 路径。
#    在 systemd scope 中，这个路径可以通过 /proc/self/cgroup 找到。
#    对于 cgroup v2，输出格式是 "0::/system.slice/your-unit.scope"
CGROUP_SLICE=$(cat /proc/self/cgroup | cut -d: -f3)
CGROUP_PATH="/sys/fs/cgroup${CGROUP_SLICE}"

# 2. 检查路径是否存在，并设置 swappiness
if [ -d "$CGROUP_PATH" ]; then
    echo "Setting swappiness=200 in cgroup: $CGROUP_PATH"
    # 使用 sudo tee 来确保有权限写入
    echo 200 | sudo tee "${CGROUP_PATH}/memory.swappiness" > /dev/null
    if [ $? -ne 0 ]; then
        echo "Error: Failed to set swappiness. Continuing without it." >&2
    fi
else
    echo "Warning: Cgroup path not found: $CGROUP_PATH" >&2
fi

# 3. 执行传入的所有参数作为真正的命令
echo "Executing command: $@"
exec "$@"
