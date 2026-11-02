cmake_minimum_required(VERSION 3.0.2)
project(tiago_remote)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/remote_server.py
  scripts/fake_signs.py
  scripts/face_detector.py
  scripts/sim_face_detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY web launch models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
