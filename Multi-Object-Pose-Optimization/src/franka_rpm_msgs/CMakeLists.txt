cmake_minimum_required(VERSION 3.4)
project(franka_rpm_msgs)

find_package(catkin REQUIRED COMPONENTS message_generation std_msgs actionlib_msgs)

add_message_files(FILES FrankaState.msg)

generate_messages(DEPENDENCIES std_msgs actionlib_msgs)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs actionlib_msgs)
