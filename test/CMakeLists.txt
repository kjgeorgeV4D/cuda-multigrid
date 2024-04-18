add_executable(test_grid test_grid.cu)
add_test(NAME test_grid COMMAND test_grid)

add_executable(test_poisson test_poisson.cu)
add_test(NAME test_poisson COMMAND test_poisson)

add_executable(heat_demo heat_demo.cu)
add_test(NAME heat_demo COMMAND heat_demo)

