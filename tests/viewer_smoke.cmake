# Exercise asynchronous scene replacement and screenshot output without a desktop or GPU.
file(MAKE_DIRECTORY "${OUTPUT}")
file(REMOVE "${OUTPUT}/00_the_lens_Eye.png" "${OUTPUT}/01_the_prism_Eye.png")
execute_process(COMMAND "${CMAKE_COMMAND}" -E env SDL_VIDEODRIVER=dummy SDL_RENDER_DRIVER=software
  "${OWE}" view "${SOURCE}/scenes/the_lens.owe" "${SOURCE}/scenes/the_prism.owe"
  --backend cpu --scale 0.1 --tour --screenshot "${OUTPUT}/" --after 0
  WORKING_DIRECTORY "${SOURCE}" RESULT_VARIABLE result TIMEOUT 45)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Viewer tour failed: ${result}")
endif()
foreach(name 00_the_lens_Eye 01_the_prism_Eye)
  file(READ "${OUTPUT}/${name}.png" signature LIMIT 8 HEX)
  if(NOT signature STREQUAL "89504e470d0a1a0a")
    message(FATAL_ERROR "Missing PNG for ${name}")
  endif()
endforeach()
execute_process(COMMAND "${OWE}" view "${SOURCE}/scenes/the_lens.owe" --view the_lens.owe:Missing
  RESULT_VARIABLE result TIMEOUT 10)
if(NOT result EQUAL 2)
  message(FATAL_ERROR "Missing detector should fail before opening a window")
endif()
