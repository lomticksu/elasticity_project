set(TARGET "seq_program")

set(TARGET_SRC
  ${TARGET}.cpp
  )

cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.7.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

deal_ii_initialize_cached_variables()
project(${TARGET})

deal_ii_invoke_autopilot()