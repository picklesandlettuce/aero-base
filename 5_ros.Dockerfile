ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV ROS_PACKAGE="ros_base" \
    ROS_VERSION="humble" \
    ROS_DISTRO="humble" \
    ROS_ROOT="/opt/ros/humble"
# Build from source is needed to enable GPU accelereated OpenCV which we installed in
# The multimedia layer
RUN --mount=type=bind,source=docker-aux/ros,target=/tmp/ros \
    /tmp/ros/ros2_build.sh

COPY --chmod=755 docker-aux/ros/ros_entrypoint.sh /ros_entrypoint.sh
COPY --chmod=755 docker-aux/ros/ros_environment.sh /ros_environment.sh
ENTRYPOINT ["/ros_entrypoint.sh"]