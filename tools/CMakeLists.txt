add_executable(vectis-cpuinfo cpuinfo.cpp)
target_link_libraries(vectis-cpuinfo PRIVATE vectis::vectis)

# Returns 1 when the compiled tier is not executable here, which is a genuine
# failure signal, so register it rather than only running it by hand.
add_test(NAME vectis_cpuinfo COMMAND vectis-cpuinfo)
