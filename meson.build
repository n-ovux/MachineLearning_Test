project('ml_test', 'c')


cc = meson.get_compiler('c')
dependencies = [cc.find_library('m'), cc.find_library('gsl')]

sources = ['./src/main.c', './src/util.c']

executable('ml_test', sources, dependencies: dependencies)
